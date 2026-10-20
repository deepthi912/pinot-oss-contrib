/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.pinot.plugin.query.gpu;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import javax.annotation.Nullable;
import org.apache.pinot.common.utils.DataSchema;
import org.apache.pinot.common.utils.DataSchema.ColumnDataType;
import org.apache.pinot.core.common.Operator;
import org.apache.pinot.core.data.table.IntermediateRecord;
import org.apache.pinot.core.data.table.Key;
import org.apache.pinot.core.data.table.Record;
import org.apache.pinot.core.operator.BaseOperator;
import org.apache.pinot.core.operator.ExecutionStatistics;
import org.apache.pinot.core.operator.blocks.results.GroupByResultsBlock;
import org.apache.pinot.core.operator.filter.BaseFilterOperator;
import org.apache.pinot.core.operator.filter.BitmapBasedFilterOperator;
import org.apache.pinot.core.operator.filter.MatchAllFilterOperator;
import org.apache.pinot.core.operator.query.GroupByOperator;
import org.apache.pinot.core.query.aggregation.function.AggregationFunction;
import org.apache.pinot.core.query.aggregation.function.AggregationFunctionUtils;
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.core.util.trace.TraceContext;
import org.apache.pinot.segment.spi.IndexSegment;
import org.apache.pinot.segment.spi.SegmentContext;
import org.apache.pinot.segment.spi.datasource.DataSource;
import org.apache.pinot.segment.spi.index.reader.Dictionary;
import org.apache.pinot.segment.spi.index.reader.ForwardIndexReader;
import org.apache.pinot.segment.spi.index.reader.ForwardIndexReaderContext;
import org.apache.pinot.spi.exception.QueryException;
import org.apache.pinot.spi.query.QueryThreadContext;
import org.roaringbitmap.BatchIterator;
import org.roaringbitmap.buffer.ImmutableRoaringBitmap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/// Segment-level `SELECT col, COUNT(*) ... GROUP BY col` on a GPU backend.
///
/// 1. CPU: Pinot's filter operator (indexes) produces the matching documents as a bitmap.
/// 2. GPU: the backend counts matching documents per dictionary id over the column's resident dictionary ids.
/// 3. CPU: non-empty groups become [IntermediateRecord]s keyed by the dictionary value, exactly as the CPU
///    group-by produces them, so the regular combine operator and broker reduce handle them unchanged.
///
/// The segment runs on the regular CPU [GroupByOperator] instead, reusing the filter result, when the filter is
/// selective, when the column cannot be resident within the GPU memory budget, or when the backend fails or returns
/// counts that do not add up to the matching documents (logged as a fallback). Cancellation and timeouts propagate.
/// Not thread-safe; like every segment operator it is used by one thread.
public class GpuGroupByOperator extends BaseOperator<GroupByResultsBlock> {
  private static final Logger LOGGER = LoggerFactory.getLogger(GpuGroupByOperator.class);
  private static final String EXPLAIN_NAME = "GPU_GROUP_BY";
  private static final int READ_CHUNK_SIZE = 10_000;
  /// Segments where fewer than 1 in this many documents match the filter run on the CPU.
  static final int SELECTIVE_FILTER_RATIO = 16;

  private final SegmentContext _segmentContext;
  private final IndexSegment _indexSegment;
  private final QueryContext _queryContext;
  private final BaseFilterOperator _filterOperator;
  private final GpuBackend _backend;
  private final GpuColumnCache _columnCache;
  private final GpuQueryStats _stats;
  private final String _groupByColumn;
  private final DataSchema _dataSchema;

  private int _numDocsScanned;
  private long _numEntriesScannedInFilter;
  private long _deviceTimeNs = -1;
  @Nullable
  private GroupByOperator _cpuOperator;

  public GpuGroupByOperator(SegmentContext segmentContext, QueryContext queryContext,
      BaseFilterOperator filterOperator, GpuBackend backend, GpuColumnCache columnCache, GpuQueryStats stats) {
    _segmentContext = segmentContext;
    _indexSegment = segmentContext.getIndexSegment();
    _queryContext = queryContext;
    _filterOperator = filterOperator;
    _backend = backend;
    _columnCache = columnCache;
    _stats = stats;
    _groupByColumn = queryContext.getGroupByExpressions().get(0).getIdentifier();

    // Same layout as GroupByOperator: group-by column first, then one column per aggregation.
    AggregationFunction[] functions = queryContext.getAggregationFunctions();
    String[] columnNames = new String[1 + functions.length];
    ColumnDataType[] columnDataTypes = new ColumnDataType[1 + functions.length];
    columnNames[0] = _groupByColumn;
    columnDataTypes[0] = ColumnDataType.fromDataTypeSV(
        _indexSegment.getDataSource(_groupByColumn).getDataSourceMetadata().getDataType());
    for (int i = 0; i < functions.length; i++) {
      columnNames[i + 1] = functions[i].getResultColumnName();
      columnDataTypes[i + 1] = functions[i].getIntermediateResultColumnType();
    }
    _dataSchema = new DataSchema(columnNames, columnDataTypes);
  }

  @Override
  protected GroupByResultsBlock getNextBlock() {
    QueryThreadContext.checkTerminationAndSampleUsage(EXPLAIN_NAME);
    int numDocs = _indexSegment.getSegmentMetadata().getTotalDocs();
    // Filtering stays on the CPU (indexes); its result is reused by every path below.
    BaseFilterOperator.FilteredDocIds filteredDocIds = _filterOperator.getFilteredDocIds();
    _numEntriesScannedInFilter = filteredDocIds.getNumEntriesScannedInFilter();
    ImmutableRoaringBitmap matchingDocs = filteredDocIds.getDocIds();
    _numDocsScanned = matchingDocs != null ? matchingDocs.getCardinality() : numDocs;
    if (_numDocsScanned == 0) {
      _stats.recordNoMatchingDocs();
      return new GroupByResultsBlock(_dataSchema, List.of(), _queryContext);
    }
    // The GPU kernel touches every document of the segment, so a selective filter is cheaper on the CPU.
    if (matchingDocs != null && (long) _numDocsScanned * SELECTIVE_FILTER_RATIO < numDocs) {
      _stats.recordSelectiveOnCpu();
      return computeOnCpu(matchingDocs, numDocs);
    }

    DataSource dataSource = _indexSegment.getDataSource(_groupByColumn);
    Dictionary dictionary = dataSource.getDictionary();
    int cardinality = dictionary.length();
    String cacheKey = GpuColumnCache.key(_backend.getType(), _queryContext.getTableName(),
        _indexSegment.getSegmentName(), _columnCache.instanceId(_indexSegment), _groupByColumn);
    long[] counts;
    try {
      GpuColumnCache.Lease lease = _columnCache.acquire(cacheKey, (long) numDocs * Integer.BYTES,
          () -> _backend.uploadIntColumn(readDictIds(dataSource, numDocs), numDocs));
      if (lease == null) {
        // Cannot be resident within the GPU memory budget: loading it for one query costs more than it saves.
        _stats.recordNotResident();
        return computeOnCpu(matchingDocs, numDocs);
      }
      GpuGroupByCountResult result;
      try (lease) {
        if (lease.getColumn().getNumValues() != numDocs) {
          throw new IllegalStateException("Resident column has " + lease.getColumn().getNumValues()
              + " values, segment has " + numDocs + " documents");
        }
        result = _backend.groupByCount(lease.getColumn(), cardinality,
            matchingDocs != null ? toWords(matchingDocs, numDocs) : null);
      }
      counts = result.getCounts();
      // Every matching document lands in exactly one group; anything else means stale or corrupt device data.
      long total = 0;
      for (long count : counts) {
        total += count;
      }
      if (total != _numDocsScanned) {
        throw new IllegalStateException("GPU counted " + total + " documents, filter matched " + _numDocsScanned);
      }
      _deviceTimeNs = result.getDeviceTimeNs();
    } catch (QueryException e) {
      // Cancellation, timeout and early termination are not GPU failures.
      throw e;
    } catch (RuntimeException e) {
      LOGGER.warn("GPU backend: {} failed on segment: {}, recomputing on CPU", _backend.getName(),
          _indexSegment.getSegmentName(), e);
      _stats.recordFallback();
      traceExecution(-1, true, e.getMessage());
      return computeOnCpu(matchingDocs, numDocs);
    }
    _stats.recordGpuExecution(_numDocsScanned, _deviceTimeNs);
    traceExecution(_deviceTimeNs, false, null);

    // Dictionaries of immutable segments are sorted, so walking dictionary ids in order yields groups sorted by value.
    // When the query orders by the group key and the sorted combine is used, emit groups in ORDER BY order, trimmed to
    // the segment trim size, as the CPU group-by does; otherwise the order does not matter.
    boolean sorted = _queryContext.shouldSortAggregateUnderSafeTrim();
    boolean ascending = !sorted || _queryContext.getOrderByExpressions().get(0).isAsc();
    int maxGroups = sorted && _queryContext.getEffectiveSegmentGroupTrimSize() > 0
        ? _queryContext.getEffectiveSegmentGroupTrimSize() : Integer.MAX_VALUE;
    // Every aggregation is COUNT(*) (see GpuQueryEligibility), so each gets the same count.
    int numAggregations = _queryContext.getAggregationFunctions().length;
    List<IntermediateRecord> records = new ArrayList<>();
    for (int i = 0; i < cardinality && records.size() < maxGroups; i++) {
      int dictId = ascending ? i : cardinality - 1 - i;
      long count = counts[dictId];
      if (count != 0) {
        Object value = dictionary.getInternal(dictId);
        Object[] row = new Object[1 + numAggregations];
        row[0] = value;
        for (int j = 1; j <= numAggregations; j++) {
          row[j] = count;
        }
        records.add(IntermediateRecord.withoutOrderByValues(new Key(new Object[]{value}), new Record(row)));
      }
    }
    GroupByResultsBlock resultsBlock = new GroupByResultsBlock(_dataSchema, records, _queryContext);
    resultsBlock.setNumGroupsWarningLimitReached(records.size() >= _queryContext.getNumGroupsWarningLimit());
    return resultsBlock;
  }

  /// Runs the regular CPU group-by over the documents the filter already matched, without filtering again.
  private GroupByResultsBlock computeOnCpu(@Nullable ImmutableRoaringBitmap matchingDocs, int numDocs) {
    BaseFilterOperator filter = matchingDocs != null ? new BitmapBasedFilterOperator(matchingDocs, false, numDocs)
        : new MatchAllFilterOperator(numDocs);
    AggregationFunctionUtils.AggregationInfo aggregationInfo =
        AggregationFunctionUtils.buildAggregationInfoWithoutStarTree(_segmentContext, _queryContext,
            _queryContext.getAggregationFunctions(), filter);
    _cpuOperator = new GroupByOperator(_queryContext, aggregationInfo, numDocs);
    return _cpuOperator.nextBlock();
  }

  /// With `trace=true`, records where this segment actually executed, so clients can verify GPU execution rather
  /// than just the plan: the backend, the device-measured kernel time (positive only when the GPU ran the work), the
  /// documents aggregated, and whether the segment fell back to the CPU.
  private void traceExecution(long deviceTimeNs, boolean fallback, @Nullable String fallbackReason) {
    if (!TraceContext.traceEnabled()) {
      return;
    }
    StringBuilder info = new StringBuilder("backend=").append(_backend.getName())
        .append(", segment=").append(_indexSegment.getSegmentName())
        .append(", docs=").append(_numDocsScanned)
        .append(", gpuKernelMs=")
        .append(deviceTimeNs >= 0 ? String.format(Locale.ROOT, "%.3f", deviceTimeNs / 1e6) : "n/a")
        .append(", fallbackToCpu=").append(fallback);
    if (fallbackReason != null) {
      info.append(", reason=").append(fallbackReason);
    }
    TraceContext.logInfo(EXPLAIN_NAME, info.toString());
  }

  /// Device-measured execution time of the last GPU call in nanoseconds, or `-1` if not measured or not run.
  public long getDeviceTimeNs() {
    return _deviceTimeNs;
  }

  /// Reads the whole column's dictionary ids once; the result stays resident on the GPU across queries.
  @SuppressWarnings("unchecked")
  private static int[] readDictIds(DataSource dataSource, int numDocs) {
    ForwardIndexReader<ForwardIndexReaderContext> reader =
        (ForwardIndexReader<ForwardIndexReaderContext>) dataSource.getForwardIndex();
    int[] dictIds = new int[numDocs];
    int[] docIds = new int[READ_CHUNK_SIZE];
    int[] buffer = new int[READ_CHUNK_SIZE];
    ForwardIndexReaderContext context = reader.createContext();
    try {
      for (int start = 0; start < numDocs; start += READ_CHUNK_SIZE) {
        int length = Math.min(READ_CHUNK_SIZE, numDocs - start);
        for (int i = 0; i < length; i++) {
          docIds[i] = start + i;
        }
        reader.readDictIds(docIds, length, buffer, context);
        System.arraycopy(buffer, 0, dictIds, start, length);
      }
    } finally {
      if (context != null) {
        context.close();
      }
    }
    return dictIds;
  }

  /// Converts the matching documents to the backend's bitmap format in batches rather than one callback per document.
  private static int[] toWords(ImmutableRoaringBitmap docIds, int numDocs) {
    int[] words = new int[(numDocs + 31) >>> 5];
    BatchIterator iterator = docIds.getBatchIterator();
    int[] batch = new int[4096];
    int n;
    while ((n = iterator.nextBatch(batch)) > 0) {
      for (int i = 0; i < n; i++) {
        int docId = batch[i];
        words[docId >>> 5] |= 1 << docId;
      }
    }
    return words;
  }

  @Override
  public List<Operator> getChildOperators() {
    return List.of(_filterOperator);
  }

  @Override
  public IndexSegment getIndexSegment() {
    return _indexSegment;
  }

  @Override
  public ExecutionStatistics getExecutionStatistics() {
    long totalDocs = _indexSegment.getSegmentMetadata().getTotalDocs();
    if (_cpuOperator != null) {
      // The CPU operator re-read the filter result from a bitmap; the filter cost was paid by _filterOperator.
      ExecutionStatistics cpu = _cpuOperator.getExecutionStatistics();
      return new ExecutionStatistics(cpu.getNumDocsScanned(), _numEntriesScannedInFilter,
          cpu.getNumEntriesScannedPostFilter(), totalDocs);
    }
    return new ExecutionStatistics(_numDocsScanned, _numEntriesScannedInFilter, _numDocsScanned, totalDocs);
  }

  @Override
  public String toExplainString() {
    StringBuilder builder = new StringBuilder(EXPLAIN_NAME).append("(backend:").append(_backend.getName())
        .append(", groupKeys:").append(_groupByColumn).append(", aggregations:");
    AggregationFunction[] functions = _queryContext.getAggregationFunctions();
    for (int i = 0; i < functions.length; i++) {
      if (i > 0) {
        builder.append(", ");
      }
      builder.append(functions[i].toExplainString());
    }
    return builder.append(')').toString();
  }
}

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
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import org.apache.pinot.common.request.context.ExpressionContext;
import org.apache.pinot.core.operator.filter.BaseFilterOperator;
import org.apache.pinot.core.query.aggregation.function.AggregationFunction;
import org.apache.pinot.core.query.aggregation.function.BaseDistinctAggregateAggregationFunction;
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.segment.local.customobject.AvgPair;
import org.apache.pinot.segment.spi.AggregationFunctionType;
import org.apache.pinot.segment.spi.IndexSegment;
import org.apache.pinot.segment.spi.datasource.DataSource;
import org.apache.pinot.segment.spi.index.reader.Dictionary;
import org.apache.pinot.segment.spi.index.reader.ForwardIndexReader;
import org.apache.pinot.segment.spi.index.reader.ForwardIndexReaderContext;
import org.apache.pinot.spi.exception.QueryException;
import org.apache.pinot.spi.query.QueryThreadContext;
import org.roaringbitmap.PeekableCharIterator;
import org.roaringbitmap.buffer.ImmutableRoaringBitmap;
import org.roaringbitmap.buffer.MappeableBitmapContainer;
import org.roaringbitmap.buffer.MappeableContainer;
import org.roaringbitmap.buffer.MappeableContainerPointer;
import org.roaringbitmap.buffer.MappeableRunContainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/// Computes one segment's COUNT / SUM / AVG / DISTINCTCOUNT on the GPU, optionally grouped by dictionary columns,
/// producing the same intermediate results as the CPU aggregation functions. [#compute()] returns `null` when the
/// segment runs on the CPU instead: no matching documents, a selective filter, a DISTINCTCOUNT bitmap too sparse or too
/// large, a multi-column group key much sparser than the matching documents, a column the cache could not make resident
/// (budget or wait), a device or backend failure (including a closed backend), or inconsistent device results. Each
/// ordered tuple of several GROUP BY columns gets its own resident combined key column. Not thread-safe; used by one
/// segment operator.
final class GpuSegmentAggregator {
  private static final Logger LOGGER = LoggerFactory.getLogger(GpuSegmentAggregator.class);
  static final int SELECTIVE_FILTER_RATIO = 16;
  // Per-group counts and sums are sized by the multi-column key's cardinality product, not by the matching documents;
  // beyond this many groups per document the CPU's hash table is cheaper.
  static final int MAX_GROUPS_PER_DOC = 4;
  private static final int READ_CHUNK_SIZE = 10_000;
  // Bounds how long a segment waits for load slots and for other threads' loads of its columns (also at most a quarter
  // of the remaining query time); a load the segment runs itself is not time-bounded. Once any lease is refused, the
  // query's remaining segments do not wait at all.
  static final long MAX_LOAD_WAIT_MS = 200;

  /// Per-group results. Values are materialized on demand, so trimmed-away groups cost nothing.
  static final class Result {
    final int _numGroups;
    final long[] _counts;
    private final AggregationFunction<?, ?>[] _functions;
    private final long[][] _sums;
    private final GpuDistinctResult[] _distinct;
    private final Dictionary[] _dictionaries;

    Result(int numGroups, long[] counts, AggregationFunction<?, ?>[] functions, long[][] sums,
        GpuDistinctResult[] distinct, Dictionary[] dictionaries) {
      _numGroups = numGroups;
      _counts = counts;
      _functions = functions;
      _sums = sums;
      _distinct = distinct;
      _dictionaries = dictionaries;
    }

    /// The intermediate result of aggregation function `function` for `group`, typed as the CPU function's.
    Object getValue(int function, int group) {
      switch (_functions[function].getType()) {
        case COUNT:
          return _counts[group];
        case SUM:
          return (double) _sums[function][group];
        case AVG:
          return new AvgPair((double) _sums[function][group], _counts[group]);
        case DISTINCTCOUNT: {
          Dictionary dictionary = _dictionaries[function];
          return BaseDistinctAggregateAggregationFunction.convertToValueSet(dictionary,
              _distinct[function].getIds(group, dictionary.length()));
        }
        default:
          throw new IllegalStateException("Unsupported aggregation: " + _functions[function].getType());
      }
    }
  }

  private final IndexSegment _segment;
  private final QueryContext _queryContext;
  private final BaseFilterOperator _filterOperator;
  private final GpuBackend _backend;
  private final GpuColumnCache _cache;
  private final GpuQueryStats _stats;
  @Nullable
  private final String[] _groupByColumns;
  @Nullable
  private final GpuGroupKeys _groupKeys;

  private int _numDocsScanned;
  private long _loadWaitDeadlineNanos;
  private long _numEntriesScannedInFilter;
  @Nullable
  private ImmutableRoaringBitmap _matchingDocs;

  GpuSegmentAggregator(IndexSegment segment, QueryContext queryContext, BaseFilterOperator filterOperator,
      GpuBackend backend, GpuColumnCache cache, GpuQueryStats stats) {
    _segment = segment;
    _queryContext = queryContext;
    _filterOperator = filterOperator;
    _backend = backend;
    _cache = cache;
    _stats = stats;
    List<ExpressionContext> groupByExpressions = queryContext.getGroupByExpressions();
    if (groupByExpressions != null) {
      _groupByColumns = new String[groupByExpressions.size()];
      for (int i = 0; i < _groupByColumns.length; i++) {
        _groupByColumns[i] = groupByExpressions.get(i).getIdentifier();
      }
      _groupKeys = new GpuGroupKeys(GpuGroupKeys.cardinalities(segment, _groupByColumns));
    } else {
      _groupByColumns = null;
      _groupKeys = null;
    }
  }

  /// The dense key of the GROUP BY columns, `null` without GROUP BY.
  @Nullable
  GpuGroupKeys getGroupKeys() {
    return _groupKeys;
  }

  int getNumDocsScanned() {
    return _numDocsScanned;
  }

  long getNumEntriesScannedInFilter() {
    return _numEntriesScannedInFilter;
  }

  /// The filtered documents, available after [#compute()]; `null` means all documents match.
  @Nullable
  ImmutableRoaringBitmap getMatchingDocs() {
    return _matchingDocs;
  }

  /// Columns read per matching document, matching the CPU projection for scan statistics. Without GROUP BY and with a
  /// filter matching every document, core answers DISTINCTCOUNT on a dictionary column from the dictionary and does
  /// not project its column (see `AggregationPlanNode`), so neither is it counted here.
  int getNumColumnsProjected() {
    Set<String> columns = new LinkedHashSet<>();
    if (_groupByColumns != null) {
      columns.addAll(Arrays.asList(_groupByColumns));
    }
    boolean distinctFromDictionary = _groupByColumns == null && _filterOperator.isResultMatchingAll();
    for (AggregationFunction<?, ?> function : _queryContext.getAggregationFunctions()) {
      if (distinctFromDictionary && function.getType() == AggregationFunctionType.DISTINCTCOUNT) {
        continue;
      }
      if (!function.getInputExpressions().isEmpty()) {
        columns.add(function.getInputExpressions().get(0).getIdentifier());
      }
    }
    return columns.size();
  }

  /// Returns the GPU result, or `null` to run the segment on the CPU over [#getMatchingDocs()].
  @Nullable
  Result compute() {
    QueryThreadContext.checkTerminationAndSampleUsage("GpuSegmentAggregator");
    int numDocs = _segment.getSegmentMetadata().getTotalDocs();
    BaseFilterOperator.FilteredDocIds filteredDocIds = _filterOperator.getFilteredDocIds();
    _numEntriesScannedInFilter = filteredDocIds.getNumEntriesScannedInFilter();
    _matchingDocs = filteredDocIds.getDocIds();
    _numDocsScanned = _matchingDocs != null ? _matchingDocs.getCardinality() : numDocs;
    if (_numDocsScanned == 0) {
      _stats.recordNoMatchingDocs();
      return null;
    }
    if (_matchingDocs != null && (long) _numDocsScanned * SELECTIVE_FILTER_RATIO < numDocs) {
      _stats.recordSelectiveOnCpu();
      return null;
    }
    if (distinctBitmapTooSparse() || combinedKeyTooSparse()) {
      _stats.recordSelectiveOnCpu();
      return null;
    }
    List<GpuColumnCache.Lease> leases = new ArrayList<>();
    try {
      Result result = computeOnDevice(numDocs, leases);
      if (result == null) {
        _stats.recordNotResident();
      }
      return result;
    } catch (QueryException e) {
      throw e;
    } catch (InconsistentResultException e) {
      LOGGER.error("GPU backend: {} returned inconsistent results on segment: {}, recomputing on CPU",
          _backend.getName(), _segment.getSegmentName(), e);
      _stats.recordFallback();
      return null;
    } catch (RuntimeException e) {
      LOGGER.warn("GPU backend: {} failed on segment: {}, recomputing on CPU", _backend.getName(),
          _segment.getSegmentName(), e);
      _stats.recordFallback();
      return null;
    } finally {
      for (GpuColumnCache.Lease lease : leases) {
        lease.close();
      }
    }
  }

  @Nullable
  private Result computeOnDevice(int numDocs, List<GpuColumnCache.Lease> leases) {
    long remainingMs = _queryContext.getEndTimeMs() - System.currentTimeMillis();
    _loadWaitDeadlineNanos = System.nanoTime() + (_stats.isLoadWaitExpired() ? 0
        : TimeUnit.MILLISECONDS.toNanos(Math.max(0, Math.min(MAX_LOAD_WAIT_MS, remainingMs / 4))));
    GpuColumn keys = null;
    int numGroups = 1;
    if (_groupKeys != null) {
      numGroups = _groupKeys.getNumGroups();
      keys = _groupByColumns.length == 1 ? residentColumn(_groupByColumns[0], false, numDocs, leases)
          : residentCombinedKeys(numDocs, leases);
      if (keys == null) {
        return null;
      }
    }
    AggregationFunction<?, ?>[] functions = _queryContext.getAggregationFunctions();
    int numFunctions = functions.length;
    String[] columns = new String[numFunctions];
    GpuColumn[] inputs = new GpuColumn[numFunctions];
    for (int i = 0; i < numFunctions; i++) {
      AggregationFunctionType type = functions[i].getType();
      if (type == AggregationFunctionType.COUNT) {
        continue;
      }
      columns[i] = functions[i].getInputExpressions().get(0).getIdentifier();
      inputs[i] = residentColumn(columns[i], type != AggregationFunctionType.DISTINCTCOUNT, numDocs, leases);
      if (inputs[i] == null) {
        return null;
      }
    }

    GpuColumn filter = _matchingDocs != null ? _backend.uploadFilter(toWords(_matchingDocs, numDocs)) : null;
    try {
      long[] counts = null;
      long deviceTimeNs = 0;
      long[][] sums = new long[numFunctions][];
      GpuDistinctResult[] distinct = new GpuDistinctResult[numFunctions];
      Dictionary[] dictionaries = new Dictionary[numFunctions];
      Map<String, long[]> sumsByColumn = new HashMap<>();
      Map<String, GpuDistinctResult> distinctByColumn = new HashMap<>();
      for (int i = 0; i < numFunctions; i++) {
        AggregationFunctionType type = functions[i].getType();
        if (type == AggregationFunctionType.SUM || type == AggregationFunctionType.AVG) {
          long[] columnSums = sumsByColumn.get(columns[i]);
          if (columnSums == null) {
            GpuAggregationResult result = _backend.aggregate(keys, numGroups, inputs[i], numDocs, filter);
            deviceTimeNs += Math.max(0, result.getDeviceTimeNs());
            // Every SUM pass also counts: the first one provides COUNT, the others must agree with it.
            if (counts == null) {
              counts = validateCounts(result, numGroups);
            } else if (!Arrays.equals(result.getCounts(), counts)) {
              throw new InconsistentResultException("Inconsistent GPU counts for column: " + columns[i]);
            }
            columnSums = result.getSums();
            if (columnSums == null || columnSums.length != numGroups) {
              throw new InconsistentResultException("Missing GPU sums for column: " + columns[i]);
            }
            sumsByColumn.put(columns[i], columnSums);
          }
          sums[i] = columnSums;
        }
      }
      if (counts == null) {
        GpuAggregationResult result = _backend.aggregate(keys, numGroups, null, numDocs, filter);
        deviceTimeNs += Math.max(0, result.getDeviceTimeNs());
        counts = validateCounts(result, numGroups);
      }
      for (int i = 0; i < numFunctions; i++) {
        if (functions[i].getType() == AggregationFunctionType.DISTINCTCOUNT) {
          dictionaries[i] = _segment.getDataSource(columns[i]).getDictionary();
          GpuDistinctResult columnDistinct = distinctByColumn.get(columns[i]);
          if (columnDistinct == null) {
            int cardinality = dictionaries[i].length();
            columnDistinct = _backend.distinct(keys, numGroups, inputs[i], cardinality, numDocs, filter);
            deviceTimeNs += Math.max(0, columnDistinct.getDeviceTimeNs());
            validateDistinct(columnDistinct, counts, cardinality, columns[i]);
            distinctByColumn.put(columns[i], columnDistinct);
          }
          distinct[i] = columnDistinct;
        }
      }
      _stats.recordGpuExecution(_numDocsScanned, deviceTimeNs);
      return new Result(numGroups, counts, functions, sums, distinct, dictionaries);
    } finally {
      if (filter != null) {
        filter.close();
      }
    }
  }

  /// DISTINCTCOUNT's bitmap has `numGroups * cardinality` bits whatever the filter; when it has more 32-bit words than
  /// matching documents, or more bits than [GpuBackend#MAX_DISTINCT_BITS], the CPU's value sets are cheaper.
  private boolean distinctBitmapTooSparse() {
    long numGroups = _groupKeys != null ? _groupKeys.getNumGroups() : 1;
    for (AggregationFunction<?, ?> function : _queryContext.getAggregationFunctions()) {
      if (function.getType() == AggregationFunctionType.DISTINCTCOUNT) {
        String column = function.getInputExpressions().get(0).getIdentifier();
        long numBits = numGroups * _segment.getDataSource(column).getDictionary().length();
        if (numBits > GpuBackend.MAX_DISTINCT_BITS || numBits / Integer.SIZE > _numDocsScanned) {
          return true;
        }
      }
    }
    return false;
  }

  private boolean combinedKeyTooSparse() {
    return _groupByColumns != null && _groupByColumns.length > 1
        && _groupKeys.getNumGroups() > (long) MAX_GROUPS_PER_DOC * _numDocsScanned;
  }

  /// Every group with matching documents must have at least one distinct value, and no other group may have any.
  private static void validateDistinct(GpuDistinctResult distinct, long[] counts, int cardinality, String column) {
    boolean[] seen = distinct.nonEmptyGroups(counts.length, cardinality);
    for (int g = 0; g < counts.length; g++) {
      if (seen[g] != (counts[g] > 0)) {
        throw new InconsistentResultException(
            "Inconsistent GPU distinct values for column: " + column + ", group: " + g);
      }
    }
  }

  private long[] validateCounts(GpuAggregationResult result, int numGroups) {
    long[] counts = result.getCounts();
    if (counts.length != numGroups) {
      throw new InconsistentResultException("GPU returned " + counts.length + " groups, expected " + numGroups);
    }
    long total = 0;
    for (long count : counts) {
      total += count;
    }
    // Stale or corrupt device data must never reach the result: recompute on CPU.
    if (total != _numDocsScanned) {
      throw new InconsistentResultException("GPU counted " + total + " documents, filter matched " + _numDocsScanned);
    }
    return counts;
  }

  @Nullable
  private GpuColumn residentColumn(String column, boolean intValues, int numDocs,
      List<GpuColumnCache.Lease> leases) {
    DataSource dataSource = _segment.getDataSource(column);
    return acquireResident(column + (intValues ? "#values" : "#dictIds"), numDocs, leases,
        () -> uploadColumn(dataSource, intValues, numDocs));
  }

  @Nullable
  private GpuColumn residentCombinedKeys(int numDocs, List<GpuColumnCache.Lease> leases) {
    // Length-prefixed names: unambiguous whatever characters the column names contain.
    StringBuilder name = new StringBuilder();
    for (String column : _groupByColumns) {
      name.append(column.length()).append(':').append(column);
    }
    return acquireResident(name.append("#combined").toString(), numDocs, leases, () -> uploadCombinedKeys(numDocs));
  }

  @Nullable
  private GpuColumn acquireResident(String name, int numDocs, List<GpuColumnCache.Lease> leases,
      Supplier<GpuColumn> loader) {
    long instanceId = _cache.getOrAssignInstanceId(_segment);
    long maxWaitNanos = Math.max(0, _loadWaitDeadlineNanos - System.nanoTime());
    String key = GpuColumnCache.cacheKey(_backend.getType(), _queryContext.getTableName(), _segment.getSegmentName(),
        instanceId, name);
    GpuColumnCache.Lease lease = _cache.acquire(key, instanceId, _backend.residentBytes(numDocs), maxWaitNanos, loader);
    if (lease == null) {
      _stats.markLoadWaitExpired();
      return null;
    }
    leases.add(lease);
    if (lease.getColumn().getNumValues() != numDocs) {
      throw new InconsistentResultException("Resident column " + name + " has " + lease.getColumn().getNumValues()
          + " values, segment has " + numDocs + " documents");
    }
    return lease.getColumn();
  }

  /// Streams a whole column to the device once: dictionary ids, or INT values (via the dictionary or a raw index).
  @SuppressWarnings("unchecked")
  private GpuColumn uploadColumn(DataSource dataSource, boolean intValues, int numDocs) {
    ForwardIndexReader<ForwardIndexReaderContext> reader =
        (ForwardIndexReader<ForwardIndexReaderContext>) dataSource.getForwardIndex();
    Dictionary dictionary = dataSource.getDictionary();
    ForwardIndexReaderContext context = reader.createContext();
    try {
      return upload(numDocs, (docIds, n, out) -> {
        if (reader.isDictionaryEncoded()) {
          reader.readDictIds(docIds, n, out, context);
          if (intValues) {
            dictionary.readIntValues(out, n, out);
          }
        } else {
          reader.readValuesSV(docIds, n, out, context);
        }
      });
    } finally {
      if (context != null) {
        context.close();
      }
    }
  }

  /// Streams the combined group ids of the GROUP BY columns to the device, reading each column's dictionary ids.
  @SuppressWarnings("unchecked")
  private GpuColumn uploadCombinedKeys(int numDocs) {
    int numColumns = _groupByColumns.length;
    ForwardIndexReader<ForwardIndexReaderContext>[] readers = new ForwardIndexReader[numColumns];
    ForwardIndexReaderContext[] contexts = new ForwardIndexReaderContext[numColumns];
    try {
      for (int c = 0; c < numColumns; c++) {
        readers[c] = (ForwardIndexReader<ForwardIndexReaderContext>) _segment.getDataSource(_groupByColumns[c])
            .getForwardIndex();
        contexts[c] = readers[c].createContext();
      }
      int[][] dictIds = new int[numColumns][READ_CHUNK_SIZE];
      return upload(numDocs, (docIds, n, out) -> {
        for (int c = 0; c < numColumns; c++) {
          readers[c].readDictIds(docIds, n, dictIds[c], contexts[c]);
        }
        _groupKeys.encode(dictIds, n, out);
      });
    } finally {
      for (ForwardIndexReaderContext context : contexts) {
        if (context != null) {
          context.close();
        }
      }
    }
  }

  /// Reads the values of `n` documents into `out`.
  private interface DocReader {
    void read(int[] docIds, int n, int[] out);
  }

  /// Uploads a column of `numDocs` values, read in chunks of at most [#READ_CHUNK_SIZE] consecutive documents.
  private GpuColumn upload(int numDocs, DocReader docReader) {
    int[] docIds = new int[READ_CHUNK_SIZE];
    int[] chunk = new int[READ_CHUNK_SIZE];
    return _backend.uploadIntColumn(numDocs, (start, length, buffer) -> {
      QueryThreadContext.checkTerminationAndSampleUsage("GpuSegmentAggregator");
      int done = 0;
      while (done < length) {
        int n = Math.min(READ_CHUNK_SIZE, length - done);
        for (int i = 0; i < n; i++) {
          docIds[i] = start + done + i;
        }
        int[] out = done == 0 && n == length ? buffer : chunk;
        docReader.read(docIds, n, out);
        if (out != buffer) {
          System.arraycopy(out, 0, buffer, done, n);
        }
        done += n;
      }
    });
  }

  /// Converts the filter result to the bitmap layout of [GpuBackend#uploadFilter], one Roaring container at a time.
  static int[] toWords(ImmutableRoaringBitmap docIds, int numDocs) {
    int[] words = new int[(numDocs + 31) >>> 5];
    long[] bitmapContainerWords = null;
    MappeableContainerPointer pointer = docIds.getContainerPointer();
    for (; pointer.hasContainer(); pointer.advance()) {
      int base = pointer.key() << 16;
      MappeableContainer container = pointer.getContainer();
      if (container instanceof MappeableBitmapContainer) {
        if (bitmapContainerWords == null) {
          bitmapContainerWords = new long[1 << 10];
        } else {
          Arrays.fill(bitmapContainerWords, 0L);
        }
        ((MappeableBitmapContainer) container).orInto(bitmapContainerWords);
        int firstWord = base >>> 5;
        for (int i = 0; i < bitmapContainerWords.length; i++) {
          // Only non-zero halves are written: the last container may extend past the segment's last word.
          long bits = bitmapContainerWords[i];
          if ((int) bits != 0) {
            words[firstWord + 2 * i] = (int) bits;
          }
          if ((int) (bits >>> 32) != 0) {
            words[firstWord + 2 * i + 1] = (int) (bits >>> 32);
          }
        }
      } else if (container instanceof MappeableRunContainer) {
        MappeableRunContainer runs = (MappeableRunContainer) container;
        for (int r = 0; r < runs.numberOfRuns(); r++) {
          int start = base + runs.getValue(r);
          setRange(words, start, start + runs.getLength(r) + 1);
        }
      } else {
        PeekableCharIterator iterator = container.getCharIterator();
        while (iterator.hasNext()) {
          int docId = base + iterator.next();
          words[docId >>> 5] |= 1 << docId;
        }
      }
    }
    return words;
  }

  /// Device results that fail validation: never merged, always recomputed on the CPU.
  private static final class InconsistentResultException extends RuntimeException {
    InconsistentResultException(String message) {
      super(message);
    }
  }

  /// Sets bits `[from, to)`.
  private static void setRange(int[] words, int from, int to) {
    int firstWord = from >>> 5;
    int lastWord = (to - 1) >>> 5;
    int firstMask = -1 << from;
    int lastMask = -1 >>> -to;
    if (firstWord == lastWord) {
      words[firstWord] |= firstMask & lastMask;
    } else {
      words[firstWord] |= firstMask;
      Arrays.fill(words, firstWord + 1, lastWord, -1);
      words[lastWord] |= lastMask;
    }
  }
}

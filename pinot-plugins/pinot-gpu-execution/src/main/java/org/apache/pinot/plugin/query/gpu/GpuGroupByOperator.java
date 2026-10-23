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
import org.apache.pinot.segment.spi.IndexSegment;
import org.apache.pinot.segment.spi.SegmentContext;
import org.apache.pinot.segment.spi.index.reader.Dictionary;
import org.roaringbitmap.buffer.ImmutableRoaringBitmap;


/// Segment-level aggregation GROUP BY one dictionary column on a GPU backend ([GpuSegmentAggregator]), producing the
/// same [GroupByResultsBlock] as [GroupByOperator]; runs [GroupByOperator] over the filter result when the segment
/// is routed to the CPU. Not thread-safe; used by one thread.
public class GpuGroupByOperator extends BaseOperator<GroupByResultsBlock> {
  private static final String EXPLAIN_NAME = "GPU_GROUP_BY";

  private final SegmentContext _segmentContext;
  private final IndexSegment _indexSegment;
  private final QueryContext _queryContext;
  private final BaseFilterOperator _filterOperator;
  private final GpuBackend _backend;
  private final GpuSegmentAggregator _aggregator;
  private final String _groupByColumn;
  private final DataSchema _dataSchema;
  @Nullable
  private GroupByOperator _cpuOperator;

  public GpuGroupByOperator(SegmentContext segmentContext, QueryContext queryContext,
      BaseFilterOperator filterOperator, GpuBackend backend, GpuColumnCache columnCache, GpuQueryStats stats) {
    _segmentContext = segmentContext;
    _indexSegment = segmentContext.getIndexSegment();
    _queryContext = queryContext;
    _filterOperator = filterOperator;
    _backend = backend;
    _aggregator = new GpuSegmentAggregator(_indexSegment, queryContext, filterOperator, backend, columnCache, stats);
    _groupByColumn = queryContext.getGroupByExpressions().get(0).getIdentifier();

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
    GpuSegmentAggregator.Result result = _aggregator.compute();
    if (result == null) {
      if (_aggregator.getNumDocsScanned() == 0) {
        return new GroupByResultsBlock(_dataSchema, List.of(), _queryContext);
      }
      return computeOnCpu(_aggregator.getMatchingDocs());
    }

    // Sorted dictionaries give ORDER BY order on the group key. Under sort-aggregate (safe trim, LIMIT below the
    // threshold) this keeps the first trim-size groups like GroupByOperator. Otherwise all groups are emitted, where
    // GroupByOperator under safe trim keeps only LIMIT of them when there are more: a superset the combine trims
    // to the same result.
    Dictionary dictionary = _indexSegment.getDataSource(_groupByColumn).getDictionary();
    boolean sorted = _queryContext.shouldSortAggregateUnderSafeTrim();
    boolean ascending = !sorted || _queryContext.getOrderByExpressions().get(0).isAsc();
    int maxGroups = sorted && _queryContext.getEffectiveSegmentGroupTrimSize() > 0
        ? _queryContext.getEffectiveSegmentGroupTrimSize() : Integer.MAX_VALUE;
    int numFunctions = _queryContext.getAggregationFunctions().length;
    int numGroups = 0;
    for (long count : result._counts) {
      if (count != 0) {
        numGroups++;
      }
    }
    List<IntermediateRecord> records = new ArrayList<>(Math.min(numGroups, maxGroups));
    for (int i = 0; i < result._numGroups && records.size() < maxGroups; i++) {
      int dictId = ascending ? i : result._numGroups - 1 - i;
      if (result._counts[dictId] == 0) {
        continue;
      }
      Object key = dictionary.getInternal(dictId);
      Object[] row = new Object[1 + numFunctions];
      row[0] = key;
      for (int f = 0; f < numFunctions; f++) {
        row[1 + f] = result.getValue(f, dictId);
      }
      records.add(IntermediateRecord.withoutOrderByValues(new Key(new Object[]{key}), new Record(row)));
    }
    GroupByResultsBlock resultsBlock = new GroupByResultsBlock(_dataSchema, records, _queryContext);
    resultsBlock.setNumGroupsWarningLimitReached(numGroups >= _queryContext.getNumGroupsWarningLimit());
    return resultsBlock;
  }

  private GroupByResultsBlock computeOnCpu(@Nullable ImmutableRoaringBitmap matchingDocs) {
    int numDocs = _indexSegment.getSegmentMetadata().getTotalDocs();
    BaseFilterOperator filter = matchingDocs != null ? new BitmapBasedFilterOperator(matchingDocs, false, numDocs)
        : new MatchAllFilterOperator(numDocs);
    _cpuOperator = new GroupByOperator(_queryContext, AggregationFunctionUtils.buildAggregationInfoWithoutStarTree(
        _segmentContext, _queryContext, _queryContext.getAggregationFunctions(), filter), numDocs);
    return _cpuOperator.nextBlock();
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
      ExecutionStatistics cpu = _cpuOperator.getExecutionStatistics();
      return new ExecutionStatistics(cpu.getNumDocsScanned(), _aggregator.getNumEntriesScannedInFilter(),
          cpu.getNumEntriesScannedPostFilter(), totalDocs);
    }
    int numDocsScanned = _aggregator.getNumDocsScanned();
    return new ExecutionStatistics(numDocsScanned, _aggregator.getNumEntriesScannedInFilter(),
        (long) numDocsScanned * _aggregator.getNumColumnsProjected(), totalDocs);
  }

  @Override
  public String toExplainString() {
    return EXPLAIN_NAME + "(backend:" + _backend.getName() + ", groupKeys:" + _groupByColumn + ", aggregations:"
        + GpuAggregationOperator.explainAggregations(_queryContext) + ')';
  }
}

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
import org.apache.pinot.common.request.context.ExpressionContext;
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
import org.apache.pinot.core.operator.query.GroupByOperator;
import org.apache.pinot.core.plan.GroupByPlanNode;
import org.apache.pinot.core.query.aggregation.function.AggregationFunction;
import org.apache.pinot.core.query.aggregation.function.AggregationFunctionUtils;
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.segment.spi.IndexSegment;
import org.apache.pinot.segment.spi.SegmentContext;
import org.apache.pinot.segment.spi.index.reader.Dictionary;
import org.roaringbitmap.buffer.ImmutableRoaringBitmap;


/// Segment-level aggregation GROUP BY dictionary columns on a GPU backend ([GpuSegmentAggregator]), producing the
/// same [GroupByResultsBlock] as [GroupByOperator]. When the segment is routed to the CPU, runs [GroupByOperator] over
/// the filter result, or core's own plan when every document matches. Not thread-safe; used by one thread.
public class GpuGroupByOperator extends BaseOperator<GroupByResultsBlock> {
  private static final String EXPLAIN_NAME = "GPU_GROUP_BY";

  private final SegmentContext _segmentContext;
  private final IndexSegment _indexSegment;
  private final QueryContext _queryContext;
  private final BaseFilterOperator _filterOperator;
  private final GpuBackend _backend;
  private final GpuSegmentAggregator _aggregator;
  private final String[] _groupByColumns;
  private final DataSchema _dataSchema;
  @Nullable
  private Operator<GroupByResultsBlock> _cpuOperator;

  public GpuGroupByOperator(SegmentContext segmentContext, QueryContext queryContext,
      BaseFilterOperator filterOperator, GpuBackend backend, GpuColumnCache columnCache, GpuQueryStats stats) {
    _segmentContext = segmentContext;
    _indexSegment = segmentContext.getIndexSegment();
    _queryContext = queryContext;
    _filterOperator = filterOperator;
    _backend = backend;
    _aggregator = new GpuSegmentAggregator(_indexSegment, queryContext, filterOperator, backend, columnCache, stats);
    List<ExpressionContext> groupByExpressions = queryContext.getGroupByExpressions();
    int numKeys = groupByExpressions.size();
    _groupByColumns = new String[numKeys];

    AggregationFunction[] functions = queryContext.getAggregationFunctions();
    String[] columnNames = new String[numKeys + functions.length];
    ColumnDataType[] columnDataTypes = new ColumnDataType[numKeys + functions.length];
    for (int i = 0; i < numKeys; i++) {
      _groupByColumns[i] = groupByExpressions.get(i).getIdentifier();
      columnNames[i] = _groupByColumns[i];
      columnDataTypes[i] = ColumnDataType.fromDataTypeSV(
          _indexSegment.getDataSource(_groupByColumns[i]).getDataSourceMetadata().getDataType());
    }
    for (int i = 0; i < functions.length; i++) {
      columnNames[numKeys + i] = functions[i].getResultColumnName();
      columnDataTypes[numKeys + i] = functions[i].getIntermediateResultColumnType();
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

    // Under sort-aggregate, group ids walked in the ORDER BY direction are in ORDER BY order (GpuQueryEligibility), so
    // the first trim-size groups match GroupByOperator. Otherwise all groups are emitted: a superset the combine trims.
    GpuGroupKeys groupKeys = _aggregator.getGroupKeys();
    int numKeys = _groupByColumns.length;
    Dictionary[] dictionaries = new Dictionary[numKeys];
    for (int c = 0; c < numKeys; c++) {
      dictionaries[c] = _indexSegment.getDataSource(_groupByColumns[c]).getDictionary();
    }
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
      int groupId = ascending ? i : result._numGroups - 1 - i;
      if (result._counts[groupId] == 0) {
        continue;
      }
      Object[] keys = new Object[numKeys];
      Object[] row = new Object[numKeys + numFunctions];
      for (int c = 0; c < numKeys; c++) {
        keys[c] = dictionaries[c].getInternal(groupKeys.decodeDictId(groupId, c));
        row[c] = keys[c];
      }
      for (int f = 0; f < numFunctions; f++) {
        row[numKeys + f] = result.getValue(f, groupId);
      }
      records.add(IntermediateRecord.withoutOrderByValues(new Key(keys), new Record(row)));
    }
    GroupByResultsBlock resultsBlock = new GroupByResultsBlock(_dataSchema, records, _queryContext);
    resultsBlock.setNumGroupsWarningLimitReached(numGroups >= _queryContext.getNumGroupsWarningLimit());
    return resultsBlock;
  }

  private GroupByResultsBlock computeOnCpu(@Nullable ImmutableRoaringBitmap matchingDocs) {
    if (matchingDocs == null) {
      _cpuOperator = new GroupByPlanNode(_segmentContext, _queryContext).run();
    } else {
      int numDocs = _indexSegment.getSegmentMetadata().getTotalDocs();
      BaseFilterOperator filter = new BitmapBasedFilterOperator(matchingDocs, false, numDocs);
      _cpuOperator = new GroupByOperator(_queryContext, AggregationFunctionUtils.buildAggregationInfoWithoutStarTree(
          _segmentContext, _queryContext, _queryContext.getAggregationFunctions(), filter), numDocs);
    }
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
    return EXPLAIN_NAME + "(backend:" + _backend.getName() + ", groupKeys:" + String.join(", ", _groupByColumns)
        + ", aggregations:" + GpuAggregationOperator.explainAggregations(_queryContext) + ')';
  }
}

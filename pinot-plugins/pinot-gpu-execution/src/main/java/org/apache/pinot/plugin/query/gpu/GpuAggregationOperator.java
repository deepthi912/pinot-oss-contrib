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
import org.apache.pinot.core.common.Operator;
import org.apache.pinot.core.operator.BaseOperator;
import org.apache.pinot.core.operator.ExecutionStatistics;
import org.apache.pinot.core.operator.blocks.results.AggregationResultsBlock;
import org.apache.pinot.core.operator.filter.BaseFilterOperator;
import org.apache.pinot.core.operator.filter.BitmapBasedFilterOperator;
import org.apache.pinot.core.operator.filter.MatchAllFilterOperator;
import org.apache.pinot.core.operator.query.AggregationOperator;
import org.apache.pinot.core.query.aggregation.function.AggregationFunction;
import org.apache.pinot.core.query.aggregation.function.AggregationFunctionUtils;
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.segment.spi.IndexSegment;
import org.apache.pinot.segment.spi.SegmentContext;
import org.roaringbitmap.buffer.ImmutableRoaringBitmap;
import org.roaringbitmap.buffer.MutableRoaringBitmap;


/// Segment-level aggregation without GROUP BY on a GPU backend ([GpuSegmentAggregator]), producing the same
/// [AggregationResultsBlock] as the scanning [AggregationOperator]; runs that operator over the filter result when the
/// segment is routed to the CPU. Not thread-safe; used by one thread.
public class GpuAggregationOperator extends BaseOperator<AggregationResultsBlock> {
  private static final String EXPLAIN_NAME = "GPU_AGGREGATE";

  private final SegmentContext _segmentContext;
  private final IndexSegment _indexSegment;
  private final QueryContext _queryContext;
  private final BaseFilterOperator _filterOperator;
  private final GpuBackend _backend;
  private final GpuSegmentAggregator _aggregator;
  @Nullable
  private AggregationOperator _cpuOperator;

  public GpuAggregationOperator(SegmentContext segmentContext, QueryContext queryContext,
      BaseFilterOperator filterOperator, GpuBackend backend, GpuColumnCache columnCache, GpuQueryStats stats) {
    _segmentContext = segmentContext;
    _indexSegment = segmentContext.getIndexSegment();
    _queryContext = queryContext;
    _filterOperator = filterOperator;
    _backend = backend;
    _aggregator = new GpuSegmentAggregator(_indexSegment, queryContext, filterOperator, backend, columnCache, stats);
  }

  @Override
  protected AggregationResultsBlock getNextBlock() {
    GpuSegmentAggregator.Result result = _aggregator.compute();
    if (result == null) {
      return computeOnCpu(_aggregator.getNumDocsScanned() == 0 ? null : _aggregator.getMatchingDocs(),
          _aggregator.getNumDocsScanned() == 0);
    }
    AggregationFunction[] functions = _queryContext.getAggregationFunctions();
    List<Object> results = new ArrayList<>(functions.length);
    for (int f = 0; f < functions.length; f++) {
      results.add(result.getValue(f, 0));
    }
    return new AggregationResultsBlock(functions, results, _queryContext);
  }

  private AggregationResultsBlock computeOnCpu(@Nullable ImmutableRoaringBitmap matchingDocs, boolean noMatch) {
    int numDocs = _indexSegment.getSegmentMetadata().getTotalDocs();
    BaseFilterOperator filter;
    if (noMatch) {
      filter = new BitmapBasedFilterOperator(new MutableRoaringBitmap(), false, numDocs);
    } else if (matchingDocs != null) {
      filter = new BitmapBasedFilterOperator(matchingDocs, false, numDocs);
    } else {
      filter = new MatchAllFilterOperator(numDocs);
    }
    _cpuOperator = new AggregationOperator(_queryContext, AggregationFunctionUtils.buildAggregationInfoWithoutStarTree(
        _segmentContext, _queryContext, _queryContext.getAggregationFunctions(), filter), numDocs);
    return _cpuOperator.nextBlock();
  }

  static String explainAggregations(QueryContext queryContext) {
    StringBuilder builder = new StringBuilder();
    for (AggregationFunction<?, ?> function : queryContext.getAggregationFunctions()) {
      if (builder.length() > 0) {
        builder.append(", ");
      }
      builder.append(function.toExplainString());
    }
    return builder.toString();
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
    return EXPLAIN_NAME + "(backend:" + _backend.getName() + ", aggregations:" + explainAggregations(_queryContext)
        + ')';
  }
}

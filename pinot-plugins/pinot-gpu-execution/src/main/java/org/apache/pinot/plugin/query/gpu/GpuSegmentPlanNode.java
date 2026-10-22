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

import org.apache.pinot.core.common.Operator;
import org.apache.pinot.core.operator.filter.BaseFilterOperator;
import org.apache.pinot.core.plan.AggregationPlanNode;
import org.apache.pinot.core.plan.FilterPlanNode;
import org.apache.pinot.core.plan.GroupByPlanNode;
import org.apache.pinot.core.plan.PlanNode;
import org.apache.pinot.core.query.aggregation.function.AggregationFunction;
import org.apache.pinot.core.query.aggregation.function.AggregationFunctionUtils;
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.segment.spi.AggregationFunctionType;
import org.apache.pinot.segment.spi.SegmentContext;


/// Segment plan for a GPU-eligible aggregation. Keeps Pinot's CPU filter (indexes), and keeps Pinot's own plan when it
/// does not scan: star-tree, or (without GROUP BY) metadata and dictionary answers for an unfiltered COUNT or
/// DISTINCTCOUNT. Not thread-safe; used by one planning thread.
public class GpuSegmentPlanNode implements PlanNode {
  private final SegmentContext _segmentContext;
  private final QueryContext _queryContext;
  private final GpuBackend _backend;
  private final GpuColumnCache _columnCache;
  private final GpuQueryStats _stats;

  public GpuSegmentPlanNode(SegmentContext segmentContext, QueryContext queryContext, GpuBackend backend,
      GpuColumnCache columnCache, GpuQueryStats stats) {
    _segmentContext = segmentContext;
    _queryContext = queryContext;
    _backend = backend;
    _columnCache = columnCache;
    _stats = stats;
  }

  @Override
  public Operator<?> run() {
    boolean groupBy = _queryContext.getGroupByExpressions() != null;
    FilterPlanNode filterPlanNode = new FilterPlanNode(_segmentContext, _queryContext);
    BaseFilterOperator filterOperator = filterPlanNode.run();
    if (!groupBy && filterOperator.isResultMatchingAll() && hasMetadataAnswerableFunction()) {
      _stats.recordIneligible();
      return new AggregationPlanNode(_segmentContext, _queryContext).run();
    }
    if (AggregationFunctionUtils.buildAggregationInfoWithStarTree(_segmentContext, _queryContext,
        _queryContext.getAggregationFunctions(), _queryContext.getFilter(), filterOperator,
        filterPlanNode.getPredicateEvaluators()) != null) {
      _stats.recordIneligible();
      return groupBy ? new GroupByPlanNode(_segmentContext, _queryContext).run()
          : new AggregationPlanNode(_segmentContext, _queryContext).run();
    }
    return groupBy
        ? new GpuGroupByOperator(_segmentContext, _queryContext, filterOperator, _backend, _columnCache, _stats)
        : new GpuAggregationOperator(_segmentContext, _queryContext, filterOperator, _backend, _columnCache, _stats);
  }

  private boolean hasMetadataAnswerableFunction() {
    for (AggregationFunction<?, ?> function : _queryContext.getAggregationFunctions()) {
      AggregationFunctionType type = function.getType();
      if (type == AggregationFunctionType.COUNT || type == AggregationFunctionType.DISTINCTCOUNT) {
        return true;
      }
    }
    return false;
  }
}

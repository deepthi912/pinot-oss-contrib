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

import java.util.List;
import javax.annotation.Nullable;
import org.apache.commons.collections4.MapUtils;
import org.apache.pinot.common.request.context.ExpressionContext;
import org.apache.pinot.core.query.aggregation.function.AggregationFunction;
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.core.query.request.context.utils.QueryContextUtils;
import org.apache.pinot.segment.spi.AggregationFunctionType;
import org.apache.pinot.segment.spi.ImmutableSegment;
import org.apache.pinot.segment.spi.IndexSegment;
import org.apache.pinot.segment.spi.datasource.DataSource;
import org.apache.pinot.segment.spi.datasource.DataSourceMetadata;
import org.apache.pinot.segment.spi.index.reader.ForwardIndexReader;


/// Decides whether a query, on a given segment, can run on the GPU path. Every check returns `null` when the GPU
/// path applies and a short reason otherwise; anything not explicitly supported stays on the regular CPU plan.
///
/// Supported so far (grows one operator at a time):
/// - `SELECT col, COUNT(*) FROM t [WHERE ...] GROUP BY col [ORDER BY ...] [LIMIT n]`
/// - `col` is a single-value, dictionary-encoded column of a sealed (immutable) segment.
///
/// Stateless and thread-safe.
///
/// Aggregations without GROUP BY are excluded on purpose: Pinot answers `COUNT(*)` from segment metadata or index
/// bitmaps without scanning, which no GPU can beat.
public final class GpuQueryEligibility {
  private GpuQueryEligibility() {
  }

  /// Segment-independent checks; call after the query options have been applied to the context.
  @Nullable
  public static String checkQuery(QueryContext queryContext) {
    if (!QueryContextUtils.isAggregationQuery(queryContext)) {
      return "not an aggregation query";
    }
    List<ExpressionContext> groupByExpressions = queryContext.getGroupByExpressions();
    if (groupByExpressions == null) {
      return "aggregation without GROUP BY";
    }
    if (groupByExpressions.size() != 1 || groupByExpressions.get(0).getType() != ExpressionContext.Type.IDENTIFIER) {
      return "GROUP BY must be a single column";
    }
    AggregationFunction[] aggregationFunctions = queryContext.getAggregationFunctions();
    for (AggregationFunction function : aggregationFunctions) {
      if (!isCountStar(function)) {
        return "unsupported aggregation: " + function.getResultColumnName();
      }
    }
    if (queryContext.hasFilteredAggregations()) {
      return "filtered aggregations";
    }
    if (queryContext.isGroupingSets()) {
      return "GROUPING SETS / ROLLUP / CUBE";
    }
    if (queryContext.isNullHandlingEnabled()) {
      return "null handling enabled";
    }
    if (queryContext.isServerReturnFinalResult() || queryContext.isServerReturnFinalResultKeyUnpartitioned()) {
      return "server returns final results";
    }
    if (queryContext.isUnsafeTrim() && queryContext.getEffectiveSegmentGroupTrimSize() > 0) {
      // The CPU group-by trims each segment by ORDER BY on the aggregation; untrimmed GPU groups could differ.
      return "segment group trim on aggregation order";
    }
    if (queryContext.getLimit() == 0) {
      return "LIMIT 0";
    }
    if (MapUtils.isNotEmpty(queryContext.getExpressionOverrideHints())) {
      return "expression override hints";
    }
    return null;
  }

  /// Checks for one segment, after [#checkQuery] passed.
  @Nullable
  public static String checkSegment(IndexSegment segment, QueryContext queryContext) {
    if (!(segment instanceof ImmutableSegment)) {
      return "mutable (consuming) segment";
    }
    String column = queryContext.getGroupByExpressions().get(0).getIdentifier();
    DataSource dataSource = segment.getDataSourceNullable(column);
    if (dataSource == null) {
      return "column " + column + " missing in segment";
    }
    DataSourceMetadata metadata = dataSource.getDataSourceMetadata();
    if (!metadata.isSingleValue()) {
      return "multi-value column " + column;
    }
    ForwardIndexReader<?> forwardIndex = dataSource.getForwardIndex();
    if (dataSource.getDictionary() == null || forwardIndex == null || !forwardIndex.isDictionaryEncoded()) {
      return "column " + column + " has no dictionary";
    }
    if (!dataSource.getDictionary().isSorted()) {
      // Group order comes from dictionary order when the query orders by the group key.
      return "column " + column + " has an unsorted dictionary";
    }
    if (metadata.getCardinality() >= queryContext.getNumGroupsLimit()) {
      // Keeps results identical to the CPU path, which stops adding groups at the limit.
      return "cardinality reaches numGroupsLimit";
    }
    return null;
  }

  /// With null handling disabled (required above), `COUNT(*)` and `COUNT(col)` both count every row and expose no
  /// input expressions.
  private static boolean isCountStar(AggregationFunction function) {
    return function.getType() == AggregationFunctionType.COUNT && function.getInputExpressions().isEmpty();
  }
}

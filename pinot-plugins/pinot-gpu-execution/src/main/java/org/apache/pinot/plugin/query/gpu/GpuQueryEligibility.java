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
import org.apache.pinot.segment.spi.index.reader.ForwardIndexReader;
import org.apache.pinot.spi.data.FieldSpec;


/// Decides whether a query, on a given segment, can run on the GPU path: COUNT(*), SUM/AVG on columns with stored type
/// INT and DISTINCTCOUNT on dictionary columns, optionally grouped by one dictionary column. Every check returns `null`
/// when the GPU path applies and a short reason otherwise; anything not explicitly supported stays on the regular CPU
/// plan. Stateless.
public final class GpuQueryEligibility {
  private GpuQueryEligibility() {
  }

  @Nullable
  public static String checkQuery(QueryContext queryContext) {
    if (!QueryContextUtils.isAggregationQuery(queryContext)) {
      return "not an aggregation query";
    }
    List<ExpressionContext> groupByExpressions = queryContext.getGroupByExpressions();
    if (groupByExpressions != null && (groupByExpressions.size() != 1
        || groupByExpressions.get(0).getType() != ExpressionContext.Type.IDENTIFIER)) {
      return "GROUP BY must be a single column";
    }
    boolean hasNonCount = false;
    for (AggregationFunction<?, ?> function : queryContext.getAggregationFunctions()) {
      AggregationFunctionType type = function.getType();
      List<ExpressionContext> inputs = function.getInputExpressions();
      if (type == AggregationFunctionType.COUNT && inputs.isEmpty()) {
        continue;
      }
      boolean supported = type == AggregationFunctionType.SUM || type == AggregationFunctionType.AVG
          || type == AggregationFunctionType.DISTINCTCOUNT;
      if (!supported || inputs.size() != 1 || inputs.get(0).getType() != ExpressionContext.Type.IDENTIFIER) {
        return "unsupported aggregation: " + function.getResultColumnName();
      }
      hasNonCount = true;
    }
    if (groupByExpressions == null && !hasNonCount) {
      // Only COUNT(*) without GROUP BY: core answers it from metadata or the filter's match count, without a scan.
      return "COUNT(*) without GROUP BY";
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
    if (queryContext.isServerReturnFinalResultKeyUnpartitioned()) {
      return "server returns final results with unpartitioned keys";
    }
    if (queryContext.isUnsafeTrim() && queryContext.getEffectiveSegmentGroupTrimSize() > 0) {
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

  static final long MAX_EXACT_DOUBLE_SUM = 1L << 53;

  @Nullable
  public static String checkSegment(IndexSegment segment, QueryContext queryContext) {
    if (!(segment instanceof ImmutableSegment)) {
      return "mutable (consuming) segment";
    }
    long numGroups = 1;
    if (queryContext.getGroupByExpressions() != null) {
      String column = queryContext.getGroupByExpressions().get(0).getIdentifier();
      String reason = checkDictionaryColumn(segment, column);
      if (reason != null) {
        return reason;
      }
      DataSource dataSource = segment.getDataSource(column);
      if (!dataSource.getDictionary().isSorted()) {
        return "column " + column + " has an unsorted dictionary";
      }
      numGroups = dataSource.getDataSourceMetadata().getCardinality();
      if (numGroups >= queryContext.getNumGroupsLimit()) {
        // Keeps results identical to the CPU path, which stops adding groups at the limit.
        return "cardinality reaches numGroupsLimit";
      }
    }
    for (AggregationFunction<?, ?> function : queryContext.getAggregationFunctions()) {
      AggregationFunctionType type = function.getType();
      if (type == AggregationFunctionType.COUNT) {
        continue;
      }
      String column = function.getInputExpressions().get(0).getIdentifier();
      if (type == AggregationFunctionType.DISTINCTCOUNT) {
        String reason = checkDictionaryColumn(segment, column);
        if (reason != null) {
          return reason;
        }
        if (numGroups * segment.getDataSource(column).getDictionary().length()
            > GpuSegmentAggregator.MAX_DISTINCT_BITS) {
          return "DISTINCTCOUNT too large for the GPU";
        }
      } else {
        DataSource dataSource = segment.getDataSourceNullable(column);
        if (dataSource == null || !dataSource.getDataSourceMetadata().isSingleValue()
            || dataSource.getForwardIndex() == null) {
          return "column " + column + " is not a single-value column with a forward index";
        }
        if (dataSource.getDataSourceMetadata().getDataType().getStoredType() != FieldSpec.DataType.INT) {
          return "SUM/AVG only on INT columns";
        }
        // The GPU sums exactly; the CPU adds doubles, which round past 2^53. Below that bound both are exact.
        Comparable<?> min = dataSource.getDataSourceMetadata().getMinValue();
        Comparable<?> max = dataSource.getDataSourceMetadata().getMaxValue();
        if (!(min instanceof Number) || !(max instanceof Number)) {
          return "column " + column + " has no min/max metadata";
        }
        long maxAbs = Math.max(Math.abs(((Number) min).longValue()), Math.abs(((Number) max).longValue()));
        if (maxAbs * segment.getSegmentMetadata().getTotalDocs() >= MAX_EXACT_DOUBLE_SUM) {
          return "SUM of " + column + " may exceed 2^53";
        }
      }
    }
    return null;
  }

  @Nullable
  private static String checkDictionaryColumn(IndexSegment segment, String column) {
    DataSource dataSource = segment.getDataSourceNullable(column);
    if (dataSource == null) {
      return "column " + column + " missing in segment";
    }
    if (!dataSource.getDataSourceMetadata().isSingleValue()) {
      return "multi-value column " + column;
    }
    ForwardIndexReader<?> forwardIndex = dataSource.getForwardIndex();
    if (dataSource.getDictionary() == null || forwardIndex == null || !forwardIndex.isDictionaryEncoded()) {
      return "column " + column + " has no dictionary";
    }
    return null;
  }
}

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

import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.core.query.request.context.utils.QueryContextConverterUtils;
import org.apache.pinot.segment.spi.ImmutableSegment;
import org.apache.pinot.segment.spi.datasource.DataSource;
import org.apache.pinot.segment.spi.datasource.DataSourceMetadata;
import org.apache.pinot.segment.spi.index.reader.Dictionary;
import org.apache.pinot.segment.spi.index.reader.ForwardIndexReader;
import org.apache.pinot.spi.data.FieldSpec.DataType;
import org.testng.annotations.Test;

import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNull;


public class GpuQueryEligibilityTest {
  private static final QueryContext SUM_QUERY = QueryContextConverterUtils.getQueryContext("SELECT SUM(v) FROM t");

  @Test
  public void testSumThatMayExceedExactDoubleRangeStaysOnCpu() {
    // 2^22 documents of up to 2^31: the CPU's double accumulation could round, so only exact-safe segments qualify.
    assertEquals(GpuQueryEligibility.checkSegment(segment(1 << 22, Integer.MIN_VALUE, 5), SUM_QUERY),
        "SUM of v may exceed 2^53");
    assertEquals(GpuQueryEligibility.checkSegment(segment((1 << 22) + 1, 0, Integer.MAX_VALUE), SUM_QUERY),
        "SUM of v may exceed 2^53");
    assertNull(GpuQueryEligibility.checkSegment(segment(1 << 22, 0, Integer.MAX_VALUE), SUM_QUERY));
    assertNull(GpuQueryEligibility.checkSegment(segment(100_000_000, -1_000_000, 1_000_000), SUM_QUERY));
  }

  @Test
  public void testSumWithoutMinMaxMetadataStaysOnCpu() {
    ImmutableSegment segment = segment(1000, 0, 10);
    when(segment.getDataSourceNullable("v").getDataSourceMetadata().getMinValue()).thenReturn(null);
    assertEquals(GpuQueryEligibility.checkSegment(segment, SUM_QUERY), "column v has no min/max metadata");
  }

  @Test
  public void testSortAggregateNeedsTheGroupByColumnOrder() {
    String select = "SELECT a, b, COUNT(*) FROM t GROUP BY a, b ORDER BY ";
    assertNull(checkQuery(select + "a, b LIMIT 10"));
    assertNull(checkQuery(select + "a DESC, b DESC LIMIT 10"));
    String reason = "ORDER BY is not the GROUP BY columns in order with one direction";
    assertEquals(checkQuery(select + "a, b DESC LIMIT 10"), reason);
    assertEquals(checkQuery(select + "b, a LIMIT 10"), reason);
    // Not sort-aggregate: above its LIMIT threshold (groups emitted in ascending key order), and ORDER BY an
    // aggregation (unsafe trim).
    assertNull(checkQuery(select + "b, a LIMIT 20000"));
    assertNull(checkQuery(select + "COUNT(*) DESC LIMIT 10"));
  }

  @Test
  public void testUnsafeSegmentTrimStaysOnCpu() {
    String reason = "unsafe segment group trim (ORDER BY other than the GROUP BY columns, or HAVING)";
    for (String query : new String[]{
        "SELECT dimInt, COUNT(*) FROM t GROUP BY dimInt ORDER BY COUNT(*) DESC, dimInt LIMIT 3",
        "SELECT dimInt, COUNT(*) FROM t GROUP BY dimInt HAVING COUNT(*) > 10 ORDER BY dimInt LIMIT 3",
        "SELECT a, b, COUNT(*) FROM t GROUP BY a, b ORDER BY COUNT(*) DESC, a, b LIMIT 3",
        "SELECT a, b, COUNT(*) FROM t GROUP BY a, b HAVING COUNT(*) > 10 ORDER BY a, b LIMIT 3"}) {
      QueryContext queryContext = QueryContextConverterUtils.getQueryContext(query);
      // Without a segment trim size, every group is kept and the GPU runs it.
      assertNull(GpuQueryEligibility.checkQuery(queryContext), query);
      queryContext.setMinSegmentGroupTrimSize(10);
      assertEquals(GpuQueryEligibility.checkQuery(queryContext), reason, query);
    }
  }

  @Test
  public void testCombinedGroupKeyBounds() {
    QueryContext query = QueryContextConverterUtils.getQueryContext("SELECT a, b, COUNT(*) FROM t GROUP BY a, b");
    query.setNumGroupsLimit(Integer.MAX_VALUE);
    assertNull(GpuQueryEligibility.checkSegment(groupBySegment(1 << 10, true, 1 << 10, true), query));
    assertEquals(GpuQueryEligibility.checkSegment(groupBySegment(17, true, 61_681, true), query),
        "group key cardinality exceeds 1048576");
    // 2^16 * (2^16 + 1) overflows an int: the product must not wrap around below the bounds.
    assertEquals(GpuQueryEligibility.checkSegment(groupBySegment(1 << 16, true, (1 << 16) + 1, true), query),
        "group key cardinality reaches numGroupsLimit");
    assertEquals(GpuQueryEligibility.checkSegment(groupBySegment(10, true, 10, false), query),
        "column b has an unsorted dictionary");
    query.setNumGroupsLimit(100);
    assertEquals(GpuQueryEligibility.checkSegment(groupBySegment(10, true, 10, true), query),
        "group key cardinality reaches numGroupsLimit");
    assertNull(GpuQueryEligibility.checkSegment(groupBySegment(11, true, 9, true), query));
  }

  private static ImmutableSegment groupBySegment(int cardinalityA, boolean sortedA, int cardinalityB,
      boolean sortedB) {
    ImmutableSegment segment = mock(ImmutableSegment.class);
    dictionaryColumn(segment, "a", cardinalityA, sortedA);
    dictionaryColumn(segment, "b", cardinalityB, sortedB);
    return segment;
  }

  private static void dictionaryColumn(ImmutableSegment segment, String column, int cardinality, boolean sorted) {
    DataSource dataSource = mock(DataSource.class);
    DataSourceMetadata metadata = mock(DataSourceMetadata.class);
    when(metadata.isSingleValue()).thenReturn(true);
    when(dataSource.getDataSourceMetadata()).thenReturn(metadata);
    ForwardIndexReader<?> forwardIndex = mock(ForwardIndexReader.class);
    when(forwardIndex.isDictionaryEncoded()).thenReturn(true);
    when(dataSource.getForwardIndex()).thenReturn((ForwardIndexReader) forwardIndex);
    Dictionary dictionary = mock(Dictionary.class);
    when(dictionary.length()).thenReturn(cardinality);
    when(dictionary.isSorted()).thenReturn(sorted);
    when(dataSource.getDictionary()).thenReturn(dictionary);
    when(segment.getDataSourceNullable(column)).thenReturn(dataSource);
    when(segment.getDataSource(column)).thenReturn(dataSource);
  }

  private static String checkQuery(String query) {
    return GpuQueryEligibility.checkQuery(QueryContextConverterUtils.getQueryContext(query));
  }

  private static ImmutableSegment segment(int numDocs, int min, int max) {
    ImmutableSegment segment = mock(ImmutableSegment.class, RETURNS_DEEP_STUBS);
    when(segment.getSegmentMetadata().getTotalDocs()).thenReturn(numDocs);
    DataSource dataSource = mock(DataSource.class);
    DataSourceMetadata metadata = mock(DataSourceMetadata.class);
    when(metadata.isSingleValue()).thenReturn(true);
    when(metadata.getDataType()).thenReturn(DataType.INT);
    when(metadata.getMinValue()).thenReturn((Comparable) min);
    when(metadata.getMaxValue()).thenReturn((Comparable) max);
    when(dataSource.getDataSourceMetadata()).thenReturn(metadata);
    when(dataSource.getForwardIndex()).thenReturn((ForwardIndexReader) mock(ForwardIndexReader.class));
    when(segment.getDataSourceNullable("v")).thenReturn(dataSource);
    return segment;
  }
}

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
import org.apache.pinot.segment.spi.index.reader.ForwardIndexReader;
import org.apache.pinot.spi.data.FieldSpec.DataType;
import org.testng.annotations.Test;

import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertNull;


public class GpuQueryEligibilityTest {
  private static final QueryContext SUM_QUERY = QueryContextConverterUtils.getQueryContext("SELECT SUM(v) FROM t");

  @Test
  public void testSumThatMayExceedExactDoubleRangeStaysOnCpu() {
    // 2^22 documents of up to 2^31: the CPU's double accumulation could round, so only exact-safe segments qualify.
    assertNotNull(GpuQueryEligibility.checkSegment(segment(1 << 22, Integer.MIN_VALUE, 5), SUM_QUERY));
    assertNotNull(GpuQueryEligibility.checkSegment(segment((1 << 22) + 1, 0, Integer.MAX_VALUE), SUM_QUERY));
    assertNull(GpuQueryEligibility.checkSegment(segment(1 << 22, 0, Integer.MAX_VALUE), SUM_QUERY));
    assertNull(GpuQueryEligibility.checkSegment(segment(100_000_000, -1_000_000, 1_000_000), SUM_QUERY));
  }

  @Test
  public void testSumWithoutMinMaxMetadataStaysOnCpu() {
    ImmutableSegment segment = segment(1000, 0, 10);
    when(segment.getDataSourceNullable("v").getDataSourceMetadata().getMinValue()).thenReturn(null);
    assertEquals(GpuQueryEligibility.checkSegment(segment, SUM_QUERY), "column v has no min/max metadata");
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

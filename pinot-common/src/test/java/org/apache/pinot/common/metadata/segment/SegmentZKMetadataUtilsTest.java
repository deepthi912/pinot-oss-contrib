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
package org.apache.pinot.common.metadata.segment;

import org.apache.pinot.segment.spi.ColumnMetadata;
import org.apache.pinot.segment.spi.SegmentMetadata;
import org.apache.pinot.segment.spi.creator.SegmentVersion;
import org.apache.pinot.spi.config.table.DedupConfig;
import org.apache.pinot.spi.config.table.TableConfig;
import org.apache.pinot.spi.config.table.TableType;
import org.apache.pinot.spi.utils.builder.TableConfigBuilder;
import org.joda.time.Interval;
import org.testng.annotations.Test;

import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNull;


public class SegmentZKMetadataUtilsTest {
  private static final String REALTIME_TABLE_NAME = "testTable_REALTIME";
  private static final String SEGMENT_NAME = "testTable_segment_0";
  private static final String DEDUP_TIME_COLUMN = "dedupTime";

  @Test
  public void testGetDedupTimeColumnForTTL() {
    assertNull(SegmentZKMetadataUtils.getDedupTimeColumnForTTL(null));
    assertNull(SegmentZKMetadataUtils.getDedupTimeColumnForTTL(tableConfig(null)));
    // Dedup without metadataTTL
    assertNull(SegmentZKMetadataUtils.getDedupTimeColumnForTTL(tableConfig(dedupConfig(true, 0, DEDUP_TIME_COLUMN))));
    // Dedup disabled
    assertNull(
        SegmentZKMetadataUtils.getDedupTimeColumnForTTL(tableConfig(dedupConfig(false, 1000, DEDUP_TIME_COLUMN))));
    assertEquals(
        SegmentZKMetadataUtils.getDedupTimeColumnForTTL(tableConfig(dedupConfig(true, 1000, DEDUP_TIME_COLUMN))),
        DEDUP_TIME_COLUMN);
    // Falls back to the table time column, like the server's dedup metadata manager
    assertEquals(SegmentZKMetadataUtils.getDedupTimeColumnForTTL(tableConfig(dedupConfig(true, 1000, null))), "ts");
  }

  @Test
  public void testDedupTimeMaxIsSetOrRemoved() {
    SegmentZKMetadata segmentZKMetadata =
        SegmentZKMetadataUtils.createSegmentZKMetadata(REALTIME_TABLE_NAME, segmentMetadata(10, 5000L), "url", null,
            100, DEDUP_TIME_COLUMN);
    assertEquals(segmentZKMetadata.getDedupTimeMax(), 5000.0);

    // A refresh records the max of the new segment, even when it is lower
    SegmentZKMetadataUtils.refreshSegmentZKMetadata(REALTIME_TABLE_NAME, segmentZKMetadata, segmentMetadata(10, 3000L),
        "url", null, 100, DEDUP_TIME_COLUMN);
    assertEquals(segmentZKMetadata.getDedupTimeMax(), 3000.0);

    // A refresh without the column removes it rather than keeping the max of the replaced segment
    SegmentZKMetadataUtils.refreshSegmentZKMetadata(REALTIME_TABLE_NAME, segmentZKMetadata, segmentMetadata(10, 3000L),
        "url", null, 100);
    assertNull(segmentZKMetadata.getDedupTimeMax());

    SegmentZKMetadataUtils.updateCommittingSegmentZKMetadata(REALTIME_TABLE_NAME, segmentZKMetadata,
        segmentMetadata(10, 7000L), "url", 100, "100", DEDUP_TIME_COLUMN);
    assertEquals(segmentZKMetadata.getDedupTimeMax(), 7000.0);

    // An empty segment has no max
    SegmentZKMetadataUtils.updateCommittingSegmentZKMetadata(REALTIME_TABLE_NAME, segmentZKMetadata,
        segmentMetadata(0, 7000L), "url", 100, "100", DEDUP_TIME_COLUMN);
    assertNull(segmentZKMetadata.getDedupTimeMax());
  }

  private static TableConfig tableConfig(DedupConfig dedupConfig) {
    TableConfig tableConfig =
        new TableConfigBuilder(TableType.REALTIME).setTableName("testTable").setTimeColumnName("ts").build();
    tableConfig.setDedupConfig(dedupConfig);
    return tableConfig;
  }

  private static DedupConfig dedupConfig(boolean dedupEnabled, double metadataTTL, String dedupTimeColumn) {
    DedupConfig dedupConfig = new DedupConfig();
    dedupConfig.setDedupEnabled(dedupEnabled);
    dedupConfig.setMetadataTTL(metadataTTL);
    dedupConfig.setDedupTimeColumn(dedupTimeColumn);
    return dedupConfig;
  }

  private static SegmentMetadata segmentMetadata(int totalDocs, long maxDedupTime) {
    SegmentMetadata segmentMetadata = mock(SegmentMetadata.class);
    when(segmentMetadata.getName()).thenReturn(SEGMENT_NAME);
    when(segmentMetadata.getTotalDocs()).thenReturn(totalDocs);
    when(segmentMetadata.getTimeInterval()).thenReturn(new Interval(0, 1000));
    when(segmentMetadata.getVersion()).thenReturn(SegmentVersion.v3);
    ColumnMetadata columnMetadata = mock(ColumnMetadata.class);
    doReturn(maxDedupTime).when(columnMetadata).getMaxValue();
    when(segmentMetadata.getColumnMetadataFor(DEDUP_TIME_COLUMN)).thenReturn(columnMetadata);
    return segmentMetadata;
  }
}

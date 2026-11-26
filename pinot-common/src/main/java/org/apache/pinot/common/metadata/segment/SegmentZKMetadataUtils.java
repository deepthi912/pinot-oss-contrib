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

import com.google.common.base.Preconditions;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;
import org.apache.commons.lang3.StringUtils;
import org.apache.pinot.common.utils.LLCSegmentName;
import org.apache.pinot.segment.spi.ColumnMetadata;
import org.apache.pinot.segment.spi.SegmentMetadata;
import org.apache.pinot.segment.spi.creator.SegmentVersion;
import org.apache.pinot.segment.spi.partition.PartitionFunction;
import org.apache.pinot.segment.spi.partition.metadata.ColumnPartitionMetadata;
import org.apache.pinot.spi.config.table.DedupConfig;
import org.apache.pinot.spi.config.table.TableConfig;
import org.apache.pinot.spi.data.DateTimeFieldSpec;
import org.apache.pinot.spi.utils.CommonConstants;
import org.apache.pinot.spi.utils.builder.TableNameBuilder;
import org.joda.time.Interval;


public class SegmentZKMetadataUtils {
  private SegmentZKMetadataUtils() {
  }

  /// Creates the segment ZK metadata for a new segment, without the dedup time max.
  public static SegmentZKMetadata createSegmentZKMetadata(String tableNameWithType, SegmentMetadata segmentMetadata,
      String downloadUrl, @Nullable String crypterName, long segmentSizeInBytes) {
    return createSegmentZKMetadata(tableNameWithType, segmentMetadata, downloadUrl, crypterName, segmentSizeInBytes,
        null);
  }

  /// Creates the segment ZK metadata for a new segment. The dedup time max is recorded when `dedupTimeColumn` is set,
  /// see [#getDedupTimeColumnForTTL].
  public static SegmentZKMetadata createSegmentZKMetadata(String tableNameWithType, SegmentMetadata segmentMetadata,
      String downloadUrl, @Nullable String crypterName, long segmentSizeInBytes, @Nullable String dedupTimeColumn) {
    SegmentZKMetadata segmentZKMetadata = new SegmentZKMetadata(segmentMetadata.getName());
    updateSegmentZKMetadata(tableNameWithType, segmentZKMetadata, segmentMetadata, downloadUrl, crypterName,
        segmentSizeInBytes, null, true, dedupTimeColumn);
    return segmentZKMetadata;
  }

  /// Refreshes the segment ZK metadata for a segment being replaced, removing the dedup time max.
  public static void refreshSegmentZKMetadata(String tableNameWithType, SegmentZKMetadata segmentZKMetadata,
      SegmentMetadata segmentMetadata, String downloadUrl, @Nullable String crypterName, long segmentSizeInBytes) {
    refreshSegmentZKMetadata(tableNameWithType, segmentZKMetadata, segmentMetadata, downloadUrl, crypterName,
        segmentSizeInBytes, null);
  }

  /// Refreshes the segment ZK metadata for a segment being replaced. The dedup time max is recorded when
  /// `dedupTimeColumn` is set and removed otherwise, so a refresh never keeps the max of the replaced segment.
  public static void refreshSegmentZKMetadata(String tableNameWithType, SegmentZKMetadata segmentZKMetadata,
      SegmentMetadata segmentMetadata, String downloadUrl, @Nullable String crypterName, long segmentSizeInBytes,
      @Nullable String dedupTimeColumn) {
    updateSegmentZKMetadata(tableNameWithType, segmentZKMetadata, segmentMetadata, downloadUrl, crypterName,
        segmentSizeInBytes, null, false, dedupTimeColumn);
  }

  /// Updates the segment ZK metadata for a committing segment, removing the dedup time max.
  public static void updateCommittingSegmentZKMetadata(String realtimeTableName, SegmentZKMetadata segmentZKMetadata,
      SegmentMetadata segmentMetadata, String downloadUrl, long segmentSizeInBytes, String endOffset) {
    updateCommittingSegmentZKMetadata(realtimeTableName, segmentZKMetadata, segmentMetadata, downloadUrl,
        segmentSizeInBytes, endOffset, null);
  }

  /// Updates the segment ZK metadata for a committing segment. The dedup time max is recorded when `dedupTimeColumn`
  /// is set and removed otherwise.
  public static void updateCommittingSegmentZKMetadata(String realtimeTableName, SegmentZKMetadata segmentZKMetadata,
      SegmentMetadata segmentMetadata, String downloadUrl, long segmentSizeInBytes, String endOffset,
      @Nullable String dedupTimeColumn) {
    // The committing segment carries the partition's dedup time max up to the previous segment, see
    // PinotLLCRealtimeSegmentManager. Keep the larger one, so the newest committed segment has the partition max.
    Double previousDedupTimeMax = segmentZKMetadata.getDedupTimeMax();
    updateSegmentZKMetadata(realtimeTableName, segmentZKMetadata, segmentMetadata, downloadUrl, null,
        segmentSizeInBytes, endOffset, false, dedupTimeColumn);
    Double dedupTimeMax = segmentZKMetadata.getDedupTimeMax();
    if (dedupTimeColumn != null && previousDedupTimeMax != null
        && (dedupTimeMax == null || previousDedupTimeMax > dedupTimeMax)) {
      segmentZKMetadata.setDedupTimeMax(previousDedupTimeMax);
    }
  }

  /// Returns the column whose max is recorded as the dedup time max in the segment ZK metadata, or `null` when the
  /// table is not a dedup table with metadataTTL. Uses the same rule as the server's dedup metadata manager: the
  /// configured dedup time column, otherwise the table time column.
  @Nullable
  public static String getDedupTimeColumnForTTL(@Nullable TableConfig tableConfig) {
    if (tableConfig == null || !tableConfig.isDedupEnabled()) {
      return null;
    }
    DedupConfig dedupConfig = tableConfig.getDedupConfig();
    if (dedupConfig.getMetadataTTL() <= 0) {
      return null;
    }
    String dedupTimeColumn = dedupConfig.getDedupTimeColumn();
    return dedupTimeColumn != null ? dedupTimeColumn : tableConfig.getValidationConfig().getTimeColumnName();
  }

  public static void updateSegmentZKTimeInterval(SegmentZKMetadata segmentZKMetadata,
      DateTimeFieldSpec dateTimeFieldSpec) {
    String startTimeString = segmentZKMetadata.getRawStartTime();
    if (StringUtils.isNotEmpty(startTimeString)) {
      long updatedStartTime = dateTimeFieldSpec.getFormatSpec().fromFormatToMillis(startTimeString);
      segmentZKMetadata.setStartTime(updatedStartTime);
    }

    String endTimeString = segmentZKMetadata.getRawEndTime();
    if (StringUtils.isNotEmpty(endTimeString)) {
      long updatedEndTime = dateTimeFieldSpec.getFormatSpec().fromFormatToMillis(endTimeString);
      segmentZKMetadata.setEndTime(updatedEndTime);
    }
  }

  private static void updateSegmentZKMetadata(String tableNameWithType, SegmentZKMetadata segmentZKMetadata,
      SegmentMetadata segmentMetadata, String downloadUrl, @Nullable String crypterName, long segmentSizeInBytes,
      @Nullable String endOffset, boolean newSegment, @Nullable String dedupTimeColumn) {
    String segmentName = segmentZKMetadata.getSegmentName();

    if (endOffset != null) {
      // For committing segment

      segmentZKMetadata.setEndOffset(endOffset);
      segmentZKMetadata.setStatus(CommonConstants.Segment.Realtime.Status.DONE);
      // for committing segments, we use data CRC to replace but only if the data CRC is present for the segment
      if (segmentMetadata.getDataCrc() >= 0) {
        segmentZKMetadata.setUseDataCrc(true);
      }

      // For committing segment, use current time as start/end time if total docs is 0
      if (segmentMetadata.getTotalDocs() > 0) {
        Interval timeInterval = segmentMetadata.getTimeInterval();
        Preconditions.checkState(timeInterval != null, "Failed to find time info for table: %s, segment: %s",
            tableNameWithType, segmentName);
        segmentZKMetadata.setStartTime(timeInterval.getStartMillis());
        segmentZKMetadata.setEndTime(timeInterval.getEndMillis());
      } else {
        long now = System.currentTimeMillis();
        segmentZKMetadata.setStartTime(now);
        segmentZKMetadata.setEndTime(now);
      }
      segmentZKMetadata.setTimeUnit(TimeUnit.MILLISECONDS);
    } else {
      // For uploaded segment
      // clear the use data crc flag for uploaded segments
      // This flag should ONLY be true for segments committed from CONSUMING state
      segmentZKMetadata.setUseDataCrc(false);
      // Set segment status, start/end offset info for real-time table
      if (TableNameBuilder.isRealtimeTableResource(tableNameWithType)) {
        segmentZKMetadata.setStatus(CommonConstants.Segment.Realtime.Status.UPLOADED);

        // For new segment, start/end offset must exist if the segment name follows LLC segment name convention
        if (newSegment && LLCSegmentName.isLLCSegment(segmentMetadata.getName())) {
          Preconditions.checkArgument(
              segmentMetadata.getStartOffset() != null && segmentMetadata.getEndOffset() != null,
              "New uploaded LLC segment must have start/end offset in the segment metadata");
        }

        // NOTE:
        // - If start/end offset is available in the uploaded segment, update them in the segment ZK metadata
        // - If not, keep the existing start/end offset in the segment ZK metadata unchanged
        if (segmentMetadata.getStartOffset() != null) {
          segmentZKMetadata.setStartOffset(segmentMetadata.getStartOffset());
        }
        if (segmentMetadata.getEndOffset() != null) {
          segmentZKMetadata.setEndOffset(segmentMetadata.getEndOffset());
        }
      }

      segmentZKMetadata.setCreationTime(segmentMetadata.getIndexCreationTime());
      if (newSegment) {
        segmentZKMetadata.setPushTime(System.currentTimeMillis());
      } else {
        segmentZKMetadata.setRefreshTime(System.currentTimeMillis());
      }
      Interval timeInterval = segmentMetadata.getTimeInterval();
      if (timeInterval != null) {
        segmentZKMetadata.setStartTime(timeInterval.getStartMillis());
        segmentZKMetadata.setEndTime(timeInterval.getEndMillis());
        segmentZKMetadata.setTimeUnit(TimeUnit.MILLISECONDS);
        ColumnMetadata timeColumnMetadata = segmentMetadata.getColumnMetadataFor(segmentMetadata.getTimeColumn());
        if (isValidTimeMetadata(timeColumnMetadata)) {
          segmentZKMetadata.setRawStartTime(timeColumnMetadata.getMinValue().toString());
          segmentZKMetadata.setRawEndTime(timeColumnMetadata.getMaxValue().toString());
        }
      } else {
        segmentZKMetadata.setStartTime(-1);
        segmentZKMetadata.setEndTime(-1);
        segmentZKMetadata.setTimeUnit(null);
      }
    }

    SegmentVersion segmentVersion = segmentMetadata.getVersion();
    segmentZKMetadata.setIndexVersion(segmentVersion != null ? segmentVersion.toString() : null);
    segmentZKMetadata.setTotalDocs(segmentMetadata.getTotalDocs());
    segmentZKMetadata.setCrc(segmentMetadata.getCrc());
    segmentZKMetadata.setDataCrc(segmentMetadata.getDataCrc());
    segmentZKMetadata.setDownloadUrl(downloadUrl);
    segmentZKMetadata.setCrypterName(crypterName);
    segmentZKMetadata.setSizeInBytes(segmentSizeInBytes);
    // Always set or remove it, so a refresh never keeps the max of the segment it replaces.
    segmentZKMetadata.setDedupTimeMax(dedupTimeColumn != null ? getMaxValue(segmentMetadata, dedupTimeColumn) : null);

    // Set partition metadata
    Map<String, ColumnPartitionMetadata> columnPartitionMap = new HashMap<>();
    for (Map.Entry<String, ColumnMetadata> entry : segmentMetadata.getColumnMetadataMap().entrySet()) {
      ColumnMetadata columnMetadata = entry.getValue();
      PartitionFunction partitionFunction = columnMetadata.getPartitionFunction();
      if (partitionFunction != null) {
        ColumnPartitionMetadata columnPartitionMetadata =
            new ColumnPartitionMetadata(partitionFunction.getName(), partitionFunction.getNumPartitions(),
                columnMetadata.getPartitions(), partitionFunction.getFunctionConfig());
        columnPartitionMap.put(entry.getKey(), columnPartitionMetadata);
      }
    }
    segmentZKMetadata.setPartitionMetadata(
        !columnPartitionMap.isEmpty() ? new SegmentPartitionMetadata(columnPartitionMap) : null);

    // Update custom metadata
    // NOTE: Do not remove existing keys because they can be set by the HTTP header from the segment upload request
    Map<String, String> customMap = segmentZKMetadata.getCustomMap();
    if (customMap == null) {
      customMap = segmentMetadata.getCustomMap();
    } else {
      customMap.putAll(segmentMetadata.getCustomMap());
    }
    segmentZKMetadata.setCustomMap(customMap);
  }

  /// Returns the max value of a numeric column, read the same way as the server's dedup metadata manager, or `null` if
  /// the segment is empty or the column has no valid numeric max.
  @Nullable
  private static Double getMaxValue(SegmentMetadata segmentMetadata, String column) {
    if (segmentMetadata.getTotalDocs() <= 0) {
      return null;
    }
    ColumnMetadata columnMetadata = segmentMetadata.getColumnMetadataFor(column);
    if (columnMetadata == null || columnMetadata.isMinMaxValueInvalid()
        || !(columnMetadata.getMaxValue() instanceof Number)) {
      return null;
    }
    return ((Number) columnMetadata.getMaxValue()).doubleValue();
  }

  private static boolean isValidTimeMetadata(ColumnMetadata timeColumnMetadata) {
    return timeColumnMetadata != null && timeColumnMetadata.getMinValue() != null
        && timeColumnMetadata.getMaxValue() != null && !timeColumnMetadata.isMinMaxValueInvalid();
  }
}

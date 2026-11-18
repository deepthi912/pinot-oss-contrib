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
package org.apache.pinot.core.data.manager.offline;

import com.google.common.annotations.VisibleForTesting;
import java.io.IOException;
import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.pinot.common.metadata.segment.SegmentZKMetadata;
import org.apache.pinot.core.data.manager.BaseTableDataManager;
import org.apache.pinot.segment.local.data.manager.SegmentDataManager;
import org.apache.pinot.segment.local.segment.index.loader.IndexLoadingConfig;
import org.apache.pinot.segment.local.upsert.PartitionUpsertMetadataManager;
import org.apache.pinot.segment.local.upsert.TableUpsertMetadataManagerFactory;
import org.apache.pinot.segment.local.utils.tablestate.TableStateUtils;
import org.apache.pinot.segment.spi.ImmutableSegment;
import org.apache.pinot.spi.config.table.TableConfig;
import org.apache.pinot.spi.data.Schema;


/// Table data manager for OFFLINE table.
///
/// OFFLINE upsert tables have no consuming segment whose commit takes the upsert snapshot and removes the primary keys
/// out of metadata TTL, as REALTIME tables do. Instead, both are done after each segment is added or replaced, once all
/// the segments of the table are loaded on server startup.
@ThreadSafe
public class OfflineTableDataManager extends BaseTableDataManager {
  private static final long ALL_SEGMENTS_LOADED_CHECK_INTERVAL_MS = 5_000L;

  private volatile boolean _allSegmentsLoaded;
  private long _lastAllSegmentsLoadedCheckTimeMs;

  @Override
  protected void doInit() {
    Pair<TableConfig, Schema> tableConfigAndSchema = getCachedTableConfigAndSchema();
    TableConfig tableConfig = tableConfigAndSchema.getLeft();
    Schema schema = tableConfigAndSchema.getRight();
    if (tableConfig.isUpsertEnabled()) {
      _tableUpsertMetadataManager =
          TableUpsertMetadataManagerFactory.create(_instanceDataManagerConfig.getUpsertConfig(), tableConfig, schema,
              this, _segmentOperationsThrottlerSet);
    }
  }

  @Override
  protected void doStart() {
  }

  @Override
  protected void doShutdown() {
    if (_tableUpsertMetadataManager != null) {
      _tableUpsertMetadataManager.stop();
    }
    releaseAndRemoveAllSegments();
    if (_tableUpsertMetadataManager != null) {
      try {
        _tableUpsertMetadataManager.close();
      } catch (IOException e) {
        _logger.warn("Caught exception while closing upsert metadata manager", e);
      }
    }
  }

  protected void doAddOnlineSegment(String segmentName)
      throws Exception {
    SegmentZKMetadata zkMetadata = fetchZKMetadata(segmentName);
    IndexLoadingConfig indexLoadingConfig = getCachedIndexLoadingConfig().withSegmentTier(zkMetadata.getTier());
    handleUpsertPreload(zkMetadata, indexLoadingConfig);
    SegmentDataManager segmentDataManager = _segmentDataManagerMap.get(segmentName);
    if (segmentDataManager == null) {
      addNewOnlineSegment(zkMetadata, indexLoadingConfig);
    } else {
      replaceSegmentIfCrcMismatch(segmentDataManager, zkMetadata, indexLoadingConfig);
    }
  }

  @Override
  protected void doAddSegment(ImmutableSegment immutableSegment, @Nullable SegmentZKMetadata zkMetadata) {
    if (isUpsertEnabled()) {
      handleUpsert(immutableSegment, zkMetadata);
      return;
    }
    super.doAddSegment(immutableSegment, zkMetadata);
  }

  /// Takes the upsert snapshot and removes the expired primary keys in the same order as a REALTIME segment commit (see
  /// RealtimeSegmentDataManager). Both are no-op when snapshot or TTL is disabled. Failures are not rethrown, as the
  /// segment is already added.
  @Override
  protected void onUpsertSegmentAdded(PartitionUpsertMetadataManager partitionUpsertMetadataManager) {
    if (!isAllSegmentsLoaded()) {
      _logger.info("Skip taking upsert snapshot before all segments are loaded");
      return;
    }
    try {
      if (partitionUpsertMetadataManager.getContext().getMetadataTTL() > 0) {
        // Take the snapshot first to persist the final state of the keys before they are removed for metadata TTL
        partitionUpsertMetadataManager.takeSnapshot();
        partitionUpsertMetadataManager.removeExpiredPrimaryKeys();
      } else {
        // Remove the deleted keys out of TTL first so that the snapshot reflects their removed valid docs right away,
        // instead of on the next segment upload
        partitionUpsertMetadataManager.removeExpiredPrimaryKeys();
        partitionUpsertMetadataManager.takeSnapshot();
      }
    } catch (Exception e) {
      _logger.warn("Caught exception while taking upsert snapshot or removing expired primary keys", e);
    }
  }

  /// Returns whether all the segments of the table were loaded at some point, checking at most once per
  /// [#ALL_SEGMENTS_LOADED_CHECK_INTERVAL_MS] until they are.
  @VisibleForTesting
  boolean isAllSegmentsLoaded() {
    if (_allSegmentsLoaded) {
      return true;
    }
    synchronized (this) {
      long currentTimeMs = System.currentTimeMillis();
      if (_allSegmentsLoaded
          || currentTimeMs - _lastAllSegmentsLoadedCheckTimeMs <= ALL_SEGMENTS_LOADED_CHECK_INTERVAL_MS) {
        return _allSegmentsLoaded;
      }
      _lastAllSegmentsLoadedCheckTimeMs = currentTimeMs;
      _allSegmentsLoaded = TableStateUtils.isAllSegmentsLoaded(_helixManager, _tableNameWithType);
      return _allSegmentsLoaded;
    }
  }

  @Override
  public void addConsumingSegment(String segmentName) {
    throw new UnsupportedOperationException("Cannot add CONSUMING segment to OFFLINE table");
  }
}

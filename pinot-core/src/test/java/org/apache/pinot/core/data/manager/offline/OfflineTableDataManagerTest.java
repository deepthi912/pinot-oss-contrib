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

import org.apache.pinot.segment.local.upsert.PartitionUpsertMetadataManager;
import org.apache.pinot.segment.local.upsert.UpsertContext;
import org.mockito.InOrder;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;


/// Tests that [OfflineTableDataManager] takes the upsert snapshot and removes the expired primary keys after a segment
/// is added to an upsert partition, in the same order as a REALTIME segment commit.
public class OfflineTableDataManagerTest {

  @Test
  public void testSnapshotBeforeRemovingExpiredKeysWithMetadataTTL() {
    PartitionUpsertMetadataManager partitionManager = mockPartitionManager(30);
    newTableDataManager(true).onUpsertSegmentAdded(partitionManager);
    InOrder inOrder = inOrder(partitionManager);
    inOrder.verify(partitionManager).takeSnapshot();
    inOrder.verify(partitionManager).removeExpiredPrimaryKeys();
  }

  @Test
  public void testRemoveExpiredKeysBeforeSnapshotWithoutMetadataTTL() {
    PartitionUpsertMetadataManager partitionManager = mockPartitionManager(0);
    newTableDataManager(true).onUpsertSegmentAdded(partitionManager);
    InOrder inOrder = inOrder(partitionManager);
    inOrder.verify(partitionManager).removeExpiredPrimaryKeys();
    inOrder.verify(partitionManager).takeSnapshot();
  }

  @Test
  public void testSkipBeforeAllSegmentsLoaded() {
    PartitionUpsertMetadataManager partitionManager = mockPartitionManager(30);
    newTableDataManager(false).onUpsertSegmentAdded(partitionManager);
    verify(partitionManager, never()).takeSnapshot();
    verify(partitionManager, never()).removeExpiredPrimaryKeys();
  }

  /// The segment is already added, so a failure must not fail the segment state transition.
  @Test
  public void testFailureIsNotRethrown() {
    PartitionUpsertMetadataManager partitionManager = mockPartitionManager(30);
    doThrow(new RuntimeException("test")).when(partitionManager).removeExpiredPrimaryKeys();
    newTableDataManager(true).onUpsertSegmentAdded(partitionManager);
    verify(partitionManager).takeSnapshot();
  }

  private static PartitionUpsertMetadataManager mockPartitionManager(double metadataTTL) {
    UpsertContext context = mock(UpsertContext.class);
    when(context.getMetadataTTL()).thenReturn(metadataTTL);
    PartitionUpsertMetadataManager partitionManager = mock(PartitionUpsertMetadataManager.class);
    when(partitionManager.getContext()).thenReturn(context);
    return partitionManager;
  }

  private static OfflineTableDataManager newTableDataManager(boolean allSegmentsLoaded) {
    return new OfflineTableDataManager() {
      {
        _logger = LoggerFactory.getLogger(OfflineTableDataManagerTest.class);
      }

      @Override
      boolean isAllSegmentsLoaded() {
        return allSegmentsLoaded;
      }
    };
  }
}

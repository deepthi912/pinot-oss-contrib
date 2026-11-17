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
package org.apache.pinot.segment.local.dedup;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.pinot.common.metadata.segment.SegmentZKMetadata;
import org.apache.pinot.common.utils.LLCSegmentName;
import org.apache.pinot.segment.local.data.manager.TableDataManager;
import org.apache.pinot.segment.local.indexsegment.immutable.ImmutableSegmentImpl;
import org.apache.pinot.segment.local.segment.readers.PinotSegmentColumnReader;
import org.apache.pinot.segment.local.segment.readers.PrimaryKeyReader;
import org.apache.pinot.segment.local.utils.HashUtils;
import org.apache.pinot.segment.local.utils.WatermarkUtils;
import org.apache.pinot.segment.spi.ColumnMetadata;
import org.apache.pinot.segment.spi.IndexSegment;
import org.apache.pinot.segment.spi.index.metadata.SegmentMetadataImpl;
import org.apache.pinot.spi.config.table.HashFunction;
import org.apache.pinot.spi.config.table.TableConfig;
import org.apache.pinot.spi.data.Schema;
import org.apache.pinot.spi.data.readers.PrimaryKey;
import org.apache.pinot.spi.utils.CommonConstants.Segment.Realtime.Status;
import org.mockito.Mockito;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;


public class ConcurrentMapPartitionDedupMetadataManagerWithTTLTest {
  private static final File TEMP_DIR = new File(FileUtils.getTempDirectory(),
      ConcurrentMapPartitionDedupMetadataManagerWithTTLTest.class.getSimpleName());
  private static final int METADATA_TTL = 10000;
  private static final String DEDUP_TIME_COLUMN_NAME = "dedupTimeColumn";
  private DedupContext.Builder _dedupContextBuilder;

  @BeforeMethod
  public void setUpContextBuilder()
      throws IOException {
    FileUtils.forceMkdir(TEMP_DIR);
    TableDataManager tableDataManager = mock(TableDataManager.class);
    when(tableDataManager.getTableDataDir()).thenReturn(TEMP_DIR);
    _dedupContextBuilder = new DedupContext.Builder()
        .setTableConfig(mock(TableConfig.class))
        .setSchema(mock(Schema.class))
        .setTableDataManager(tableDataManager)
        .setPrimaryKeyColumns(List.of("primaryKeyColumn"))
        .setMetadataTTL(METADATA_TTL)
        .setDedupTimeColumn(DEDUP_TIME_COLUMN_NAME);
  }

  @AfterMethod
  public void cleanup() {
    FileUtils.deleteQuietly(TEMP_DIR);
  }

  @Test
  public void creatingMetadataManagerThrowsExceptions() {
    TableDataManager tableDataManager = mock(TableDataManager.class);
    when(tableDataManager.getTableDataDir()).thenReturn(TEMP_DIR);
    DedupContext dedupContext = new DedupContext.Builder()
        .setTableConfig(mock(TableConfig.class))
        .setSchema(mock(Schema.class))
        .setTableDataManager(tableDataManager)
        .setPrimaryKeyColumns(List.of("primaryKeyColumn"))
        .setHashFunction(HashFunction.NONE)
        .setMetadataTTL(1)
        .setDedupTimeColumn(null)
        .build();
    assertThrows(IllegalArgumentException.class,
        () -> new ConcurrentMapPartitionDedupMetadataManager(DedupTestUtils.REALTIME_TABLE_NAME, 0, dedupContext));
  }

  @Test
  public void testRemoveExpiredPrimaryKeys()
      throws IOException {
    verifyRemoveExpiredPrimaryKeys(HashFunction.NONE);
    verifyRemoveExpiredPrimaryKeys(HashFunction.MD5);
    verifyRemoveExpiredPrimaryKeys(HashFunction.MURMUR3);
  }

  private void verifyRemoveExpiredPrimaryKeys(HashFunction hashFunction)
      throws IOException {
    _dedupContextBuilder.setHashFunction(hashFunction);
    ConcurrentMapPartitionDedupMetadataManager metadataManager =
        new ConcurrentMapPartitionDedupMetadataManager(DedupTestUtils.REALTIME_TABLE_NAME, 0,
            _dedupContextBuilder.build());

    IndexSegment segment = Mockito.mock(IndexSegment.class);
    for (int i = 0; i < 20; i++) {
      double time = i * 1000;
      Object primaryKeyKey = HashUtils.hashPrimaryKey(DedupTestUtils.getPrimaryKey(i), hashFunction);
      metadataManager._primaryKeyToSegmentAndTimeMap.computeIfAbsent(primaryKeyKey, k -> Pair.of(segment, time));
    }
    metadataManager._largestSeenTime.set(19000);
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.size(), 20);
    verifyInMemoryState(metadataManager, 0, 20, segment, hashFunction);

    metadataManager.removeExpiredPrimaryKeys();
    assertEquals(metadataManager.getNumPrimaryKeys(), 11);
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.size(), 11);
    verifyInMemoryState(metadataManager, 9, 11, segment, hashFunction);

    metadataManager.stop();
    metadataManager.close();
  }

  @Test
  public void testAddRemoveTheSameSegment()
      throws IOException {
    verifyAddRemoveTheSameSegment(HashFunction.NONE);
    verifyAddRemoveTheSameSegment(HashFunction.MD5);
    verifyAddRemoveTheSameSegment(HashFunction.MURMUR3);
  }

  private void verifyAddRemoveTheSameSegment(HashFunction hashFunction)
      throws IOException {
    _dedupContextBuilder.setHashFunction(hashFunction);
    ConcurrentMapPartitionDedupMetadataManager metadataManager =
        new ConcurrentMapPartitionDedupMetadataManager(DedupTestUtils.REALTIME_TABLE_NAME, 0,
            _dedupContextBuilder.build());

    DedupUtils.DedupRecordInfoReader dedupRecordInfoReader = generateDedupRecordInfoReader(10, 0);
    Iterator<DedupRecordInfo> dedupRecordInfoIterator =
        DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader, 10);
    IndexSegment segment = DedupTestUtils.mockSegment(1, 10);
    metadataManager.doAddOrReplaceSegment(null, segment, dedupRecordInfoIterator);
    verifyInitialSegmentAddition(metadataManager, segment, hashFunction);

    dedupRecordInfoIterator = DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader, 10);
    metadataManager.doRemoveSegment(segment, dedupRecordInfoIterator);
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.size(), 0);

    metadataManager.stop();
    metadataManager.close();
  }

  @Test
  public void verifyAddingTwoSegmentWithSamePrimaryKeys()
      throws IOException {
    verifyAddingTwoSegmentWithSamePrimaryKeys(HashFunction.NONE);
    verifyAddingTwoSegmentWithSamePrimaryKeys(HashFunction.MD5);
    verifyAddingTwoSegmentWithSamePrimaryKeys(HashFunction.MURMUR3);
  }

  private void verifyAddingTwoSegmentWithSamePrimaryKeys(HashFunction hashFunction)
      throws IOException {
    _dedupContextBuilder.setHashFunction(hashFunction);
    ConcurrentMapPartitionDedupMetadataManager metadataManager =
        new ConcurrentMapPartitionDedupMetadataManager(DedupTestUtils.REALTIME_TABLE_NAME, 0,
            _dedupContextBuilder.build());

    DedupUtils.DedupRecordInfoReader dedupRecordInfoReader = generateDedupRecordInfoReader(10, 0);
    IndexSegment segment = DedupTestUtils.mockSegment(1, 10);
    Iterator<DedupRecordInfo> dedupRecordInfoIterator =
        DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader, 10);
    metadataManager.doAddOrReplaceSegment(null, segment, dedupRecordInfoIterator);
    verifyInitialSegmentAddition(metadataManager, segment, hashFunction);

    IndexSegment segment2 = DedupTestUtils.mockSegment(2, 10);
    dedupRecordInfoIterator = DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader, 10);
    metadataManager.doAddOrReplaceSegment(segment, segment2, dedupRecordInfoIterator);
    verifyInitialSegmentAddition(metadataManager, segment2, hashFunction);

    metadataManager.stop();
    metadataManager.close();
  }

  @Test
  public void testRemoveAnotherSegmentWithTheSamePrimaryKeys()
      throws IOException {
    verifyRemoveAnotherSegmentWithTheSamePrimaryKeys(HashFunction.NONE);
    verifyRemoveAnotherSegmentWithTheSamePrimaryKeys(HashFunction.MD5);
    verifyRemoveAnotherSegmentWithTheSamePrimaryKeys(HashFunction.MURMUR3);
  }

  private void verifyRemoveAnotherSegmentWithTheSamePrimaryKeys(HashFunction hashFunction)
      throws IOException {
    _dedupContextBuilder.setHashFunction(hashFunction);
    ConcurrentMapPartitionDedupMetadataManager metadataManager =
        new ConcurrentMapPartitionDedupMetadataManager(DedupTestUtils.REALTIME_TABLE_NAME, 0,
            _dedupContextBuilder.build());

    DedupUtils.DedupRecordInfoReader dedupRecordInfoReader = generateDedupRecordInfoReader(10, 0);
    IndexSegment segment = DedupTestUtils.mockSegment(1, 10);
    Iterator<DedupRecordInfo> dedupRecordInfoIterator =
        DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader, 10);
    metadataManager.doAddOrReplaceSegment(null, segment, dedupRecordInfoIterator);
    verifyInitialSegmentAddition(metadataManager, segment, hashFunction);

    IndexSegment segment2 = DedupTestUtils.mockSegment(2, 10);
    dedupRecordInfoIterator = DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader, 10);
    metadataManager.doRemoveSegment(segment2, dedupRecordInfoIterator);
    verifyInitialSegmentAddition(metadataManager, segment, hashFunction);

    metadataManager.stop();
    metadataManager.close();
  }

  private void verifyInitialSegmentAddition(ConcurrentMapPartitionDedupMetadataManager metadataManager,
      IndexSegment segment, HashFunction hashFunction) {
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.size(), 10);
    verifyInMemoryState(metadataManager, 0, 10, segment, hashFunction);
  }

  private void verifyInMemoryState(ConcurrentMapPartitionDedupMetadataManager metadataManager, int startPrimaryKeyId,
      int recordCount, IndexSegment segment, HashFunction hashFunction) {
    for (int primaryKeyId = startPrimaryKeyId; primaryKeyId < startPrimaryKeyId + recordCount; primaryKeyId++) {
      Object primaryKey = HashUtils.hashPrimaryKey(DedupTestUtils.getPrimaryKey(primaryKeyId), hashFunction);
      assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.get(primaryKey),
          Pair.of(segment, (double) primaryKeyId * 1000));
    }
  }

  @Test
  public void testAddTwoDifferentSegmentsRemoveEarlySegmentFirst()
      throws IOException {
    verifyAddTwoDifferentSegmentsRemoveEarlySegmentFirst(HashFunction.NONE);
    verifyAddTwoDifferentSegmentsRemoveEarlySegmentFirst(HashFunction.MD5);
    verifyAddTwoDifferentSegmentsRemoveEarlySegmentFirst(HashFunction.MURMUR3);
  }

  private void verifyAddTwoDifferentSegmentsRemoveEarlySegmentFirst(HashFunction hashFunction)
      throws IOException {
    // Start each hash function as a fresh server, without the watermark file the previous run persisted. Otherwise
    // the keys added below are out of TTL against that watermark and are not added.
    FileUtils.cleanDirectory(TEMP_DIR);
    _dedupContextBuilder.setHashFunction(hashFunction);
    ConcurrentMapPartitionDedupMetadataManager metadataManager =
        new ConcurrentMapPartitionDedupMetadataManager(DedupTestUtils.REALTIME_TABLE_NAME, 0,
            _dedupContextBuilder.build());

    DedupUtils.DedupRecordInfoReader dedupRecordInfoReader1 = generateDedupRecordInfoReader(10, 0);
    IndexSegment segment1 = DedupTestUtils.mockSegment(1, 10);
    Iterator<DedupRecordInfo> dedupRecordInfoIterator1 =
        DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader1, 10);
    metadataManager.doAddOrReplaceSegment(null, segment1, dedupRecordInfoIterator1);
    verifyInitialSegmentAddition(metadataManager, segment1, hashFunction);

    DedupUtils.DedupRecordInfoReader dedupRecordInfoReader2 = generateDedupRecordInfoReader(10, 10);
    IndexSegment segment2 = DedupTestUtils.mockSegment(2, 10);
    Iterator<DedupRecordInfo> dedupRecordInfoIterator2 =
        DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader2, 10);
    metadataManager.doAddOrReplaceSegment(null, segment2, dedupRecordInfoIterator2);

    metadataManager._largestSeenTime.set(19000);
    metadataManager.removeExpiredPrimaryKeys();
    assertEquals(metadataManager.getNumPrimaryKeys(), 11);
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.size(), 11);
    verifyInMemoryState(metadataManager, 9, 1, segment1, hashFunction);
    verifyInMemoryState(metadataManager, 10, 10, segment2, hashFunction);
    assertEquals(WatermarkUtils.loadWatermark(metadataManager.getWatermarkFile(), -1), 19000);

    dedupRecordInfoIterator1 = DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader1, 10);
    metadataManager.doRemoveSegment(segment1, dedupRecordInfoIterator1);
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.size(), 10);
    verifyInMemoryState(metadataManager, 10, 10, segment2, hashFunction);

    dedupRecordInfoIterator2 = DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader2, 10);
    metadataManager.doRemoveSegment(segment2, dedupRecordInfoIterator2);
    assertTrue(metadataManager._primaryKeyToSegmentAndTimeMap.isEmpty());

    metadataManager.stop();
    metadataManager.close();
  }

  @Test
  public void testAddTwoDifferentSegmentsRemoveRecentSegmentFirst()
      throws IOException {
    verifyAddTwoDifferentSegmentsRemoveRecentSegmentFirst(HashFunction.NONE);
    verifyAddTwoDifferentSegmentsRemoveRecentSegmentFirst(HashFunction.MD5);
    verifyAddTwoDifferentSegmentsRemoveRecentSegmentFirst(HashFunction.MURMUR3);
  }

  private void verifyAddTwoDifferentSegmentsRemoveRecentSegmentFirst(HashFunction hashFunction)
      throws IOException {
    // Start each hash function as a fresh server, without the watermark file the previous run persisted. Otherwise
    // the keys added below are out of TTL against that watermark and are not added.
    FileUtils.cleanDirectory(TEMP_DIR);
    _dedupContextBuilder.setHashFunction(hashFunction);
    ConcurrentMapPartitionDedupMetadataManager metadataManager =
        new ConcurrentMapPartitionDedupMetadataManager(DedupTestUtils.REALTIME_TABLE_NAME, 0,
            _dedupContextBuilder.build());

    DedupUtils.DedupRecordInfoReader dedupRecordInfoReader1 = generateDedupRecordInfoReader(10, 0);
    IndexSegment segment1 = DedupTestUtils.mockSegment(1, 10);
    Iterator<DedupRecordInfo> dedupRecordInfoIterator1 =
        DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader1, 10);
    metadataManager.doAddOrReplaceSegment(null, segment1, dedupRecordInfoIterator1);
    verifyInitialSegmentAddition(metadataManager, segment1, hashFunction);

    DedupUtils.DedupRecordInfoReader dedupRecordInfoReader2 = generateDedupRecordInfoReader(10, 10);
    IndexSegment segment2 = DedupTestUtils.mockSegment(2, 10);
    Iterator<DedupRecordInfo> dedupRecordInfoIterator2 =
        DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader2, 10);
    metadataManager.doAddOrReplaceSegment(null, segment2, dedupRecordInfoIterator2);
    metadataManager._largestSeenTime.set(19000);
    metadataManager.removeExpiredPrimaryKeys();
    assertEquals(metadataManager.getNumPrimaryKeys(), 11);
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.size(), 11);
    verifyInMemoryState(metadataManager, 10, 10, segment2, hashFunction);
    assertEquals(WatermarkUtils.loadWatermark(metadataManager.getWatermarkFile(), -1), 19000);

    dedupRecordInfoIterator2 = DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader2, 10);
    metadataManager.doRemoveSegment(segment2, dedupRecordInfoIterator2);
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.size(), 1);
    verifyInMemoryState(metadataManager, 9, 1, segment1, hashFunction);

    dedupRecordInfoIterator1 = DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader1, 10);
    metadataManager.doRemoveSegment(segment1, dedupRecordInfoIterator1);
    assertTrue(metadataManager._primaryKeyToSegmentAndTimeMap.isEmpty());

    metadataManager.stop();
    metadataManager.close();
  }

  @Test
  public void testAddingSegmentWithDuplicatedPrimaryKeys()
      throws IOException {
    verifyAddingSegmentWithDuplicatedPrimaryKeys(HashFunction.NONE);
    verifyAddingSegmentWithDuplicatedPrimaryKeys(HashFunction.MD5);
    verifyAddingSegmentWithDuplicatedPrimaryKeys(HashFunction.MURMUR3);
  }

  private void verifyAddingSegmentWithDuplicatedPrimaryKeys(HashFunction hashFunction)
      throws IOException {
    _dedupContextBuilder.setHashFunction(hashFunction);
    ConcurrentMapPartitionDedupMetadataManager metadataManager =
        new ConcurrentMapPartitionDedupMetadataManager(DedupTestUtils.REALTIME_TABLE_NAME, 0,
            _dedupContextBuilder.build());

    PrimaryKey primaryKey = DedupTestUtils.getPrimaryKey(0);
    PrimaryKeyReader primaryKeyReader = Mockito.mock(PrimaryKeyReader.class);
    for (int i = 0; i < 3; i++) {
      Mockito.when(primaryKeyReader.getPrimaryKey(i)).thenReturn(primaryKey);
    }
    PinotSegmentColumnReader dedupTimeColumnReader = Mockito.mock(PinotSegmentColumnReader.class);
    Mockito.when(dedupTimeColumnReader.getValue(0)).thenReturn(1000.0);
    Mockito.when(dedupTimeColumnReader.getValue(1)).thenReturn(15000.0);
    Mockito.when(dedupTimeColumnReader.getValue(2)).thenReturn(25000.0);
    DedupUtils.DedupRecordInfoReader dedupRecordInfoReader =
        new DedupUtils.DedupRecordInfoReader(primaryKeyReader, dedupTimeColumnReader);
    metadataManager._largestSeenTime.set(20000);

    ImmutableSegmentImpl immutableSegment = DedupTestUtils.mockSegment(1, 3);
    Iterator<DedupRecordInfo> dedupRecordInfoIterator = DedupUtils.getDedupRecordInfoIterator(dedupRecordInfoReader, 3);
    metadataManager.doAddOrReplaceSegment(null, immutableSegment, dedupRecordInfoIterator);
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.size(), 1);
    Object primaryKeyHash = HashUtils.hashPrimaryKey(primaryKey, hashFunction);
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.size(), 1);
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.get(primaryKeyHash),
        Pair.of(immutableSegment, 25000.0));

    metadataManager.stop();
    metadataManager.close();
  }

  @Test
  public void testAddRow()
      throws IOException {
    verifyAddRow(HashFunction.NONE);
    verifyAddRow(HashFunction.MD5);
    verifyAddRow(HashFunction.MURMUR3);
  }

  private void verifyAddRow(HashFunction hashFunction)
      throws IOException {
    _dedupContextBuilder.setHashFunction(hashFunction);
    ConcurrentMapPartitionDedupMetadataManager metadataManager =
        new ConcurrentMapPartitionDedupMetadataManager(DedupTestUtils.REALTIME_TABLE_NAME, 0,
            _dedupContextBuilder.build());

    metadataManager._largestSeenTime.set(20000);

    PrimaryKey primaryKey = DedupTestUtils.getPrimaryKey(0);
    DedupRecordInfo dedupRecordInfo = new DedupRecordInfo(primaryKey, 1000);
    ImmutableSegmentImpl immutableSegment = DedupTestUtils.mockSegment(1, 1);
    assertFalse(metadataManager.checkRecordPresentOrUpdate(dedupRecordInfo, immutableSegment));
    assertFalse(metadataManager._primaryKeyToSegmentAndTimeMap.isEmpty());
    assertEquals(metadataManager._largestSeenTime.get(), 20000);

    Object primaryKeyHash = HashUtils.hashPrimaryKey(primaryKey, hashFunction);
    dedupRecordInfo = new DedupRecordInfo(primaryKey, 15000);
    assertFalse(metadataManager.checkRecordPresentOrUpdate(dedupRecordInfo, immutableSegment));
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.size(), 1);
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.get(primaryKeyHash),
        Pair.of(immutableSegment, 15000.0));
    assertEquals(metadataManager._largestSeenTime.get(), 20000);

    dedupRecordInfo = new DedupRecordInfo(primaryKey, 25000);
    assertTrue(metadataManager.checkRecordPresentOrUpdate(dedupRecordInfo, immutableSegment));
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.size(), 1);
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.get(primaryKeyHash),
        Pair.of(immutableSegment, 15000.0));
    assertEquals(metadataManager._largestSeenTime.get(), 25000);

    metadataManager.stop();
    metadataManager.close();
  }

  private static DedupUtils.DedupRecordInfoReader generateDedupRecordInfoReader(int numberOfDocs,
      int startPrimaryKeyValue) {
    PrimaryKeyReader primaryKeyReader = Mockito.mock(PrimaryKeyReader.class);
    PinotSegmentColumnReader dedupTimeColumnReader = Mockito.mock(PinotSegmentColumnReader.class);
    for (int i = 0; i < numberOfDocs; i++) {
      int primaryKeyValue = startPrimaryKeyValue + i;
      Mockito.when(primaryKeyReader.getPrimaryKey(i)).thenReturn(DedupTestUtils.getPrimaryKey(primaryKeyValue));
      double time = primaryKeyValue * 1000;
      Mockito.when(dedupTimeColumnReader.getValue(i)).thenReturn(time);
    }
    return new DedupUtils.DedupRecordInfoReader(primaryKeyReader, dedupTimeColumnReader);
  }

  // Each segment below holds 10 keys: segment n has keys [10n, 10n + 9] with dedup time key * 1000, so one TTL
  // (10000) spans 10 keys. "Window" in the comments means the keys within TTL of the watermark: 11 keys, both ends
  // included.

  // A server newly assigned a partition has no watermark file, so the watermark starts at 0. Loading the history
  // oldest first used to put every key into the map; expired keys must be dropped while the segments load.
  @Test
  public void testColdBootstrapOldestFirstKeepsMapBounded()
      throws IOException {
    verifyColdBootstrapOldestFirst(false);
    verifyColdBootstrapOldestFirst(true);
  }

  private void verifyColdBootstrapOldestFirst(boolean preload)
      throws IOException {
    RowFeedingDedupMetadataManager metadataManager = newRowFeedingMetadataManager();
    ImmutableSegmentImpl[] segments = new ImmutableSegmentImpl[6];
    for (int i = 0; i < 6; i++) {
      segments[i] = metadataManager.addRows(i, 10 * i, 10);
      if (preload) {
        metadataManager.preloadSegment(segments[i]);
      } else {
        metadataManager.addSegment(segments[i]);
      }
      // One window plus the segment being loaded, never the whole history.
      assertTrue(metadataManager.getNumPrimaryKeys() <= 21,
          "Map holds " + metadataManager.getNumPrimaryKeys() + " keys after loading segment " + i);
    }
    assertEquals(metadataManager._largestSeenTime.get(), 59000.0);
    assertEquals(metadataManager.getNumPrimaryKeys(), 11);
    verifyKey(metadataManager, 49, segments[4]);
    for (int key = 50; key < 60; key++) {
      verifyKey(metadataManager, key, segments[5]);
    }
    closeMetadataManager(metadataManager);
  }

  // Loading newest first skips the older segments as a whole, and the rows of the segment straddling the TTL
  // boundary that are already expired must not be added either.
  @Test
  public void testColdBootstrapNewestFirstSkipsExpiredRows()
      throws IOException {
    RowFeedingDedupMetadataManager metadataManager = newRowFeedingMetadataManager();
    ImmutableSegmentImpl[] segments = new ImmutableSegmentImpl[6];
    for (int i = 5; i >= 0; i--) {
      segments[i] = metadataManager.addRows(i, 10 * i, 10);
      metadataManager.addSegment(segments[i]);
    }
    assertEquals(metadataManager.getNumPrimaryKeys(), 11);
    verifyKey(metadataManager, 49, segments[4]);
    for (int key = 40; key < 49; key++) {
      assertFalse(metadataManager._primaryKeyToSegmentAndTimeMap.containsKey(hash(key)));
    }
    closeMetadataManager(metadataManager);
  }

  // Restart: the watermark loaded from the file already sets the window, so loading needs no sweep. The expired
  // rows of a segment straddling the boundary are still not added.
  @Test
  public void testRestartWithPersistedWatermark()
      throws IOException {
    RowFeedingDedupMetadataManager previousRun = newRowFeedingMetadataManager();
    WatermarkUtils.persistWatermark(59000, previousRun.getWatermarkFile());
    closeMetadataManager(previousRun);

    RowFeedingDedupMetadataManager metadataManager = newRowFeedingMetadataManager();
    assertEquals(metadataManager._largestSeenTime.get(), 59000.0);
    for (int i = 0; i < 6; i++) {
      metadataManager.addSegment(metadataManager.addRows(i, 10 * i, 10));
    }
    assertEquals(metadataManager.getNumPrimaryKeys(), 11);
    assertEquals(metadataManager._numSweeps, 0);
    closeMetadataManager(metadataManager);
  }

  // Segment commit replaces the consuming segment through replaceSegment, which must keep its behavior: stale
  // entries move to the committed segment, no row is skipped and no sweep runs on the commit path.
  @Test
  public void testCommitReplaceDoesNotSkipRowsOrSweep()
      throws IOException {
    RowFeedingDedupMetadataManager metadataManager = newRowFeedingMetadataManager();
    IndexSegment consumingSegment = mock(IndexSegment.class);
    for (int key = 0; key < 30; key++) {
      metadataManager.checkRecordPresentOrUpdate(new DedupRecordInfo(DedupTestUtils.getPrimaryKey(key), key * 1000),
          consumingSegment);
    }
    // The watermark is 29000, so keys 0 to 18 are out of TTL, and it moved past TTL since the last sweep.
    ImmutableSegmentImpl committedSegment = metadataManager.addRows(1, 0, 30);
    metadataManager.replaceSegment(consumingSegment, committedSegment);
    assertEquals(metadataManager.getNumPrimaryKeys(), 30);
    for (int key = 0; key < 30; key++) {
      verifyKey(metadataManager, key, committedSegment);
    }
    assertEquals(metadataManager._numSweeps, 0);
    closeMetadataManager(metadataManager);
  }

  // TTL cleanup: the consuming thread's sweep counts as the last sweep, so loading a segment soon after it does not
  // sweep again.
  @Test
  public void testConsumingSweepCountsAsLastSweep()
      throws IOException {
    RowFeedingDedupMetadataManager metadataManager = newRowFeedingMetadataManager();
    metadataManager.checkRecordPresentOrUpdate(new DedupRecordInfo(DedupTestUtils.getPrimaryKey(50), 50000),
        mock(IndexSegment.class));
    metadataManager.removeExpiredPrimaryKeys();
    assertEquals(metadataManager._numSweeps, 1);
    // Moves the watermark from 50000 to 59000, less than TTL past the consuming sweep.
    metadataManager.addSegment(metadataManager.addRows(5, 50, 10));
    assertEquals(metadataManager._numSweeps, 1);
    // Moves the watermark to 69000, a full TTL past the consuming sweep.
    metadataManager.addSegment(metadataManager.addRows(6, 60, 10));
    assertEquals(metadataManager._numSweeps, 2);
    assertEquals(metadataManager.getNumPrimaryKeys(), 11);
    closeMetadataManager(metadataManager);
  }

  // Backfill: an uploaded segment goes through addSegment. Its expired rows are not added, and a segment that moves
  // the watermark by TTL or more sweeps right away instead of at the next consuming segment.
  @Test
  public void testBackfillSegment()
      throws IOException {
    RowFeedingDedupMetadataManager metadataManager = newRowFeedingMetadataManager();
    IndexSegment consumingSegment = mock(IndexSegment.class);
    metadataManager.checkRecordPresentOrUpdate(new DedupRecordInfo(DedupTestUtils.getPrimaryKey(59), 59000),
        consumingSegment);
    metadataManager.removeExpiredPrimaryKeys();

    // Backfill of keys 30 to 59: keys 30 to 48 are out of TTL against the watermark 59000.
    ImmutableSegmentImpl backfill = metadataManager.addRows(100, 30, 30);
    metadataManager.addSegment(backfill);
    assertEquals(metadataManager.getNumPrimaryKeys(), 11);
    for (int key = 30; key < 49; key++) {
      assertFalse(metadataManager._primaryKeyToSegmentAndTimeMap.containsKey(hash(key)));
    }

    // Backfill with newer data: the watermark moves to 109000 and the keys before 99000 are swept.
    ImmutableSegmentImpl newerBackfill = metadataManager.addRows(101, 100, 10);
    metadataManager.addSegment(newerBackfill);
    assertEquals(metadataManager._largestSeenTime.get(), 109000.0);
    assertEquals(metadataManager.getNumPrimaryKeys(), 10);
    for (int key = 100; key < 110; key++) {
      verifyKey(metadataManager, key, newerBackfill);
    }
    closeMetadataManager(metadataManager);
  }

  // Preload puts each row without comparing dedup times, and segments preload in any order. A key re-ingested after
  // its old copy expired is in two segments; if the older segment straddles the TTL boundary and preloads last, its
  // expired copy used to overwrite the live entry, so a duplicate within TTL was accepted.
  @Test
  public void testPreloadDoesNotOverwriteLiveKeyWithExpiredCopy()
      throws IOException {
    RowFeedingDedupMetadataManager metadataManager = newRowFeedingMetadataManager();
    ImmutableSegmentImpl newerSegment = metadataManager.addRows(2, 50, 10);
    metadataManager.preloadSegment(newerSegment);
    // Key 50 at 30000 is out of TTL against the watermark 59000; key 70 at 55000 keeps the segment within TTL.
    ImmutableSegmentImpl olderSegment = metadataManager.addRows(1, new int[]{50, 70}, new double[]{30000, 55000});
    metadataManager.preloadSegment(olderSegment);
    verifyKey(metadataManager, 50, newerSegment);
    assertTrue(metadataManager.checkRecordPresentOrUpdate(
        new DedupRecordInfo(DedupTestUtils.getPrimaryKey(50), 52000), mock(IndexSegment.class)));
    closeMetadataManager(metadataManager);
  }

  // A server newly assigned the partition seeds the watermark from the segment ZK metadata before loading, so the
  // old segments are skipped as a whole and the map never holds more than one window.
  @Test
  public void testSeededWatermarkSkipsOldSegments()
      throws IOException {
    RowFeedingDedupMetadataManager metadataManager = newRowFeedingMetadataManager();
    metadataManager._watermarkSeed = 59000.0;
    for (int i = 0; i < 6; i++) {
      metadataManager.addSegment(metadataManager.addRows(i, 10 * i, 10));
      assertTrue(metadataManager.getNumPrimaryKeys() <= 11,
          "Map holds " + metadataManager.getNumPrimaryKeys() + " keys after loading segment " + i);
    }
    assertEquals(metadataManager._largestSeenTime.get(), 59000.0);
    assertEquals(metadataManager.getNumPrimaryKeys(), 11);
    assertEquals(metadataManager._numSweeps, 0);
    assertEquals(metadataManager._numSeedFetches, 1);
    closeMetadataManager(metadataManager);
  }

  // The seed only raises the watermark: a higher one from the local file is kept, and a lower one left by an earlier
  // assignment of the partition is raised. The consuming thread seeds through removeExpiredPrimaryKeys.
  @Test
  public void testSeedOnlyRaisesWatermark()
      throws IOException {
    RowFeedingDedupMetadataManager previousRun = newRowFeedingMetadataManager();
    WatermarkUtils.persistWatermark(69000, previousRun.getWatermarkFile());
    closeMetadataManager(previousRun);
    RowFeedingDedupMetadataManager metadataManager = newRowFeedingMetadataManager();
    metadataManager._watermarkSeed = 59000.0;
    metadataManager.removeExpiredPrimaryKeys();
    assertEquals(metadataManager._largestSeenTime.get(), 69000.0);
    closeMetadataManager(metadataManager);

    previousRun = newRowFeedingMetadataManager();
    WatermarkUtils.persistWatermark(9000, previousRun.getWatermarkFile());
    closeMetadataManager(previousRun);
    metadataManager = newRowFeedingMetadataManager();
    metadataManager._watermarkSeed = 59000.0;
    metadataManager.removeExpiredPrimaryKeys();
    assertEquals(metadataManager._largestSeenTime.get(), 59000.0);
    closeMetadataManager(metadataManager);
  }

  // A failed seed read is logged and the segments load without it, and the read is not retried on every operation.
  @Test
  public void testSeedFailureLoadsWithoutSeed()
      throws IOException {
    RowFeedingDedupMetadataManager metadataManager = newRowFeedingMetadataManager();
    metadataManager._seedFailure = new RuntimeException("ZK unavailable");
    metadataManager.addSegment(metadataManager.addRows(5, 50, 10));
    metadataManager.addSegment(metadataManager.addRows(4, 40, 10));
    assertEquals(metadataManager._largestSeenTime.get(), 59000.0);
    assertEquals(metadataManager._numSeedFetches, 1);
    closeMetadataManager(metadataManager);
  }

  @Test
  public void testSelectWatermarkSeed()
      throws IOException {
    ConcurrentMapPartitionDedupMetadataManager metadataManager =
        new ConcurrentMapPartitionDedupMetadataManager(DedupTestUtils.REALTIME_TABLE_NAME, 0,
            _dedupContextBuilder.setHashFunction(HashFunction.NONE).build());
    Map<String, SegmentZKMetadata> zkMetadataMap = new HashMap<>();
    // Partition 0: committed segments 0 to 4 and the consuming segment 5
    for (int sequenceNumber = 0; sequenceNumber < 5; sequenceNumber++) {
      putSegmentZKMetadata(zkMetadataMap, 0, sequenceNumber, Status.DONE, sequenceNumber * 1000.0, null);
    }
    putSegmentZKMetadata(zkMetadataMap, 0, 5, Status.IN_PROGRESS, null, null);
    // Other partitions and uploaded segments are ignored, even with a higher max
    putSegmentZKMetadata(zkMetadataMap, 1, 9, Status.DONE, 99000.0, null);
    List<String> segmentNames = new ArrayList<>(zkMetadataMap.keySet());
    segmentNames.add("uploaded__testTable__0__20260101T0000Z__suffix");
    assertEquals(metadataManager.selectWatermarkSeed(segmentNames, zkMetadataMap::get), 4000.0);

    // The newest committed segment has no max (committed before the upgrade): use the one before it
    putSegmentZKMetadata(zkMetadataMap, 0, 4, Status.DONE, null, null);
    assertEquals(metadataManager.selectWatermarkSeed(segmentNames, zkMetadataMap::get), 3000.0);

    // At most MAX_SEGMENTS_TO_CHECK_FOR_WATERMARK_SEED committed segments are checked
    putSegmentZKMetadata(zkMetadataMap, 0, 3, Status.DONE, null, null);
    putSegmentZKMetadata(zkMetadataMap, 0, 2, Status.DONE, null, null);
    assertNull(metadataManager.selectWatermarkSeed(segmentNames, zkMetadataMap::get));
    assertNull(metadataManager.selectWatermarkSeed(List.of(), zkMetadataMap::get));
    closeMetadataManager(metadataManager);

    // No seed when segments on a non-default tier are not added: the partition max may come from one of them
    putSegmentZKMetadata(zkMetadataMap, 0, 4, Status.DONE, 4000.0, null);
    metadataManager = new ConcurrentMapPartitionDedupMetadataManager(DedupTestUtils.REALTIME_TABLE_NAME, 0,
        _dedupContextBuilder.setHashFunction(HashFunction.NONE).setIgnoreNonDefaultTiers(true).build());
    assertNull(metadataManager.selectWatermarkSeed(segmentNames, zkMetadataMap::get));
    closeMetadataManager(metadataManager);
  }

  // Without a Helix manager, e.g. in a minion task, there is nothing to seed from.
  @Test
  public void testNoSeedWithoutHelixManager()
      throws IOException {
    ConcurrentMapPartitionDedupMetadataManager metadataManager =
        new ConcurrentMapPartitionDedupMetadataManager(DedupTestUtils.REALTIME_TABLE_NAME, 0,
            _dedupContextBuilder.setHashFunction(HashFunction.NONE).build());
    assertNull(metadataManager.fetchWatermarkSeed());
    closeMetadataManager(metadataManager);
  }

  private static void putSegmentZKMetadata(Map<String, SegmentZKMetadata> zkMetadataMap, int partitionId,
      int sequenceNumber, Status status, Double dedupTimeMax, String tier) {
    String segmentName =
        new LLCSegmentName(DedupTestUtils.RAW_TABLE_NAME, partitionId, sequenceNumber, 0).getSegmentName();
    SegmentZKMetadata segmentZKMetadata = new SegmentZKMetadata(segmentName);
    segmentZKMetadata.setStatus(status);
    segmentZKMetadata.setDedupTimeMax(dedupTimeMax);
    segmentZKMetadata.setTier(tier);
    zkMetadataMap.put(segmentName, segmentZKMetadata);
  }

  private RowFeedingDedupMetadataManager newRowFeedingMetadataManager() {
    return new RowFeedingDedupMetadataManager(_dedupContextBuilder.setHashFunction(HashFunction.NONE).build());
  }

  private static Object hash(int key) {
    return HashUtils.hashPrimaryKey(DedupTestUtils.getPrimaryKey(key), HashFunction.NONE);
  }

  private static void verifyKey(ConcurrentMapPartitionDedupMetadataManager metadataManager, int key,
      IndexSegment segment) {
    assertEquals(metadataManager._primaryKeyToSegmentAndTimeMap.get(hash(key)), Pair.of(segment, key * 1000.0));
  }

  private static void closeMetadataManager(ConcurrentMapPartitionDedupMetadataManager metadataManager)
      throws IOException {
    metadataManager.stop();
    metadataManager.close();
  }

  /// Feeds each segment's rows to the manager and counts the expired key sweeps.
  private static class RowFeedingDedupMetadataManager extends ConcurrentMapPartitionDedupMetadataManager {
    private final Map<IndexSegment, DedupUtils.DedupRecordInfoReader> _readers = new HashMap<>();
    private int _numSweeps;
    private Double _watermarkSeed;
    private RuntimeException _seedFailure;
    private int _numSeedFetches;

    RowFeedingDedupMetadataManager(DedupContext dedupContext) {
      super(DedupTestUtils.REALTIME_TABLE_NAME, 0, dedupContext);
    }

    /// Returns a segment with keys [startKey, startKey + numDocs - 1], each with dedup time key * 1000.
    ImmutableSegmentImpl addRows(int sequenceNumber, int startKey, int numDocs) {
      int[] keys = new int[numDocs];
      double[] times = new double[numDocs];
      for (int i = 0; i < numDocs; i++) {
        keys[i] = startKey + i;
        times[i] = keys[i] * 1000.0;
      }
      return addRows(sequenceNumber, keys, times);
    }

    /// Returns a segment whose doc i has primary key keys[i] and dedup time times[i].
    ImmutableSegmentImpl addRows(int sequenceNumber, int[] keys, double[] times) {
      ImmutableSegmentImpl segment = DedupTestUtils.mockSegment(sequenceNumber, keys.length);
      ColumnMetadata columnMetadata = mock(ColumnMetadata.class);
      doReturn(Arrays.stream(times).max().orElseThrow()).when(columnMetadata).getMaxValue();
      when(segment.getSegmentMetadata().getColumnMetadataFor(DEDUP_TIME_COLUMN_NAME)).thenReturn(columnMetadata);
      PrimaryKeyReader primaryKeyReader = mock(PrimaryKeyReader.class);
      PinotSegmentColumnReader dedupTimeColumnReader = mock(PinotSegmentColumnReader.class);
      for (int i = 0; i < keys.length; i++) {
        when(primaryKeyReader.getPrimaryKey(i)).thenReturn(DedupTestUtils.getPrimaryKey(keys[i]));
        when(dedupTimeColumnReader.getValue(i)).thenReturn(times[i]);
      }
      _readers.put(segment, new DedupUtils.DedupRecordInfoReader(primaryKeyReader, dedupTimeColumnReader));
      return segment;
    }

    @Override
    DedupUtils.DedupRecordInfoReader createDedupRecordInfoReader(IndexSegment segment) {
      return _readers.get(segment);
    }

    @Override
    Double fetchWatermarkSeed() {
      _numSeedFetches++;
      if (_seedFailure != null) {
        throw _seedFailure;
      }
      return _watermarkSeed;
    }

    @Override
    protected void doRemoveExpiredPrimaryKeys() {
      _numSweeps++;
      super.doRemoveExpiredPrimaryKeys();
    }
  }

  @Test
  public void testAddSegmentAfterStop() {
    verifyAddSegmentAfterStop(HashFunction.NONE);
    verifyAddSegmentAfterStop(HashFunction.MD5);
    verifyAddSegmentAfterStop(HashFunction.MURMUR3);
  }

  // skipSegmentOutOfTTL must not bump the watermark: caller does that after the rows are added, so a concurrent
  // sweep cannot expire keys the in-flight add is about to insert.
  @Test
  public void testSkipSegmentOutOfTTLDoesNotBumpWatermark()
      throws IOException {
    _dedupContextBuilder.setHashFunction(HashFunction.NONE);
    ConcurrentMapPartitionDedupMetadataManager metadataManager =
        new ConcurrentMapPartitionDedupMetadataManager(DedupTestUtils.REALTIME_TABLE_NAME, 0,
            _dedupContextBuilder.build());

    metadataManager._largestSeenTime.set(5000);

    IndexSegment segment = DedupTestUtils.mockSegment(1, 10);
    SegmentMetadataImpl segmentMetadata = mock(SegmentMetadataImpl.class);
    ColumnMetadata columnMetadata = mock(ColumnMetadata.class);
    when(segmentMetadata.getColumnMetadataMap()).thenReturn(new TreeMap<>() {{
        this.put(DEDUP_TIME_COLUMN_NAME, columnMetadata);
      }});
    when(segmentMetadata.getColumnMetadataFor(DEDUP_TIME_COLUMN_NAME)).thenReturn(columnMetadata);
    doReturn(10000.0).when(columnMetadata).getMaxValue();
    when(segment.getSegmentMetadata()).thenReturn(segmentMetadata);

    assertFalse(metadataManager.skipSegmentOutOfTTL(segment));
    assertEquals(metadataManager._largestSeenTime.get(), 5000.0);

    metadataManager.updateLargestSeenTime(segment);
    assertEquals(metadataManager._largestSeenTime.get(), 10000.0);

    metadataManager.stop();
    metadataManager.close();
  }

  private void verifyAddSegmentAfterStop(HashFunction hashFunction) {
    _dedupContextBuilder.setHashFunction(hashFunction);
    ConcurrentMapPartitionDedupMetadataManager metadataManager =
        new ConcurrentMapPartitionDedupMetadataManager(DedupTestUtils.REALTIME_TABLE_NAME, 0,
            _dedupContextBuilder.build());

    IndexSegment segment = DedupTestUtils.mockSegment(1, 10);
    SegmentMetadataImpl segmentMetadata = mock(SegmentMetadataImpl.class);
    ColumnMetadata columnMetadata = mock(ColumnMetadata.class);
    when(segmentMetadata.getColumnMetadataMap()).thenReturn(new TreeMap<>() {{
        this.put(DEDUP_TIME_COLUMN_NAME, columnMetadata);
      }});
    when(segmentMetadata.getColumnMetadataFor(DEDUP_TIME_COLUMN_NAME)).thenReturn(columnMetadata);
    doReturn(System.currentTimeMillis()).when(columnMetadata).getMaxValue();
    when(segment.getSegmentMetadata()).thenReturn(segmentMetadata);
    // throws when not stopped
    assertThrows(RuntimeException.class, () -> metadataManager.addSegment(segment));

    metadataManager.stop();
    // do not throw when stopped
    metadataManager.addSegment(segment);
  }
}

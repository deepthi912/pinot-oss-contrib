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
package org.apache.pinot.segment.local.indexsegment.mutable;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.apache.pinot.segment.local.data.manager.TableDataManager;
import org.apache.pinot.segment.local.indexsegment.immutable.ImmutableSegmentImpl;
import org.apache.pinot.segment.local.indexsegment.immutable.ImmutableSegmentLoader;
import org.apache.pinot.segment.local.segment.creator.TransformPipeline;
import org.apache.pinot.segment.local.segment.creator.impl.SegmentIndexCreationDriverImpl;
import org.apache.pinot.segment.local.segment.readers.GenericRowRecordReader;
import org.apache.pinot.segment.local.upsert.PartitionUpsertMetadataManager;
import org.apache.pinot.segment.local.upsert.TableUpsertMetadataManager;
import org.apache.pinot.segment.local.upsert.TableUpsertMetadataManagerFactory;
import org.apache.pinot.segment.spi.creator.SegmentGeneratorConfig;
import org.apache.pinot.spi.config.table.HashFunction;
import org.apache.pinot.spi.config.table.TableConfig;
import org.apache.pinot.spi.config.table.TableType;
import org.apache.pinot.spi.config.table.UpsertConfig;
import org.apache.pinot.spi.data.FieldSpec.DataType;
import org.apache.pinot.spi.data.Schema;
import org.apache.pinot.spi.data.readers.FileFormat;
import org.apache.pinot.spi.data.readers.GenericRow;
import org.apache.pinot.spi.data.readers.RecordReader;
import org.apache.pinot.spi.data.readers.RecordReaderFactory;
import org.apache.pinot.spi.env.PinotConfiguration;
import org.apache.pinot.spi.utils.ReadMode;
import org.apache.pinot.spi.utils.builder.TableConfigBuilder;
import org.roaringbitmap.buffer.ImmutableRoaringBitmap;
import org.testng.annotations.Test;

import static org.mockito.Mockito.mock;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertTrue;


public class MutableSegmentImplUpsertTest {
  private static final String SCHEMA_FILE_PATH = "data/test_upsert_schema.json";
  private static final String DATA_FILE_PATH = "data/test_upsert_data.json";
  private static final String RAW_TABLE_NAME = "testTable";
  private static final String TIME_COLUMN = "secondsSinceEpoch";
  private static final String OTHER_COMPARISON_COLUMN = "otherComparisonColumn";
  private static final String PRIMARY_KEY_COLUMN = "event_id";

  private MutableSegmentImpl _mutableSegmentImpl;
  private PartitionUpsertMetadataManager _partitionUpsertMetadataManager;

  private UpsertConfig createPartialUpsertConfig(HashFunction hashFunction) {
    UpsertConfig upsertConfigWithHash = new UpsertConfig(UpsertConfig.Mode.PARTIAL);
    upsertConfigWithHash.setHashFunction(hashFunction);
    upsertConfigWithHash.setComparisonColumns(List.of(TIME_COLUMN, OTHER_COMPARISON_COLUMN));
    return upsertConfigWithHash;
  }

  private UpsertConfig createFullUpsertConfig(HashFunction hashFunction) {
    UpsertConfig upsertConfigWithHash = new UpsertConfig(UpsertConfig.Mode.FULL);
    upsertConfigWithHash.setHashFunction(hashFunction);
    return upsertConfigWithHash;
  }

  private void setup(UpsertConfig upsertConfigWithHash)
      throws Exception {
    TableConfig tableConfig = new TableConfigBuilder(TableType.REALTIME).setTableName(RAW_TABLE_NAME)
        .setTimeColumnName(TIME_COLUMN)
        .setUpsertConfig(upsertConfigWithHash)
        .setNullHandlingEnabled(true)
        .build();
    URL schemaResourceUrl = getClass().getClassLoader().getResource(SCHEMA_FILE_PATH);
    assertNotNull(schemaResourceUrl);
    Schema schema = Schema.fromFile(new File(schemaResourceUrl.getFile()));
    TransformPipeline transformPipeline = new TransformPipeline(tableConfig, schema);
    URL dataResourceUrl = getClass().getClassLoader().getResource(DATA_FILE_PATH);
    assertNotNull(dataResourceUrl);
    File jsonFile = new File(dataResourceUrl.getFile());
    TableUpsertMetadataManager tableUpsertMetadataManager =
        TableUpsertMetadataManagerFactory.create(new PinotConfiguration(), tableConfig, schema,
            mock(TableDataManager.class), null);
    _partitionUpsertMetadataManager = tableUpsertMetadataManager.getOrCreatePartitionManager(0);
    _mutableSegmentImpl =
        MutableSegmentImplTestUtils.createMutableSegmentImpl(schema, true, TIME_COLUMN, _partitionUpsertMetadataManager,
            null);

    GenericRow reuse = new GenericRow();
    try (RecordReader recordReader = RecordReaderFactory.getRecordReader(FileFormat.JSON, jsonFile,
        schema.getColumnNames(), null)) {
      while (recordReader.hasNext()) {
        recordReader.next(reuse);
        TransformPipeline.Result result = transformPipeline.processRow(reuse);
        for (GenericRow transformedRow : result.getTransformedRows()) {
          _mutableSegmentImpl.index(transformedRow, null);
        }
        reuse.clear();
      }
    }
  }

  private void tearDown()
      throws IOException {
    if (_mutableSegmentImpl != null) {
      _mutableSegmentImpl.destroy();
      _mutableSegmentImpl = null;
    }
    if (_partitionUpsertMetadataManager != null) {
      _partitionUpsertMetadataManager.stop();
      _partitionUpsertMetadataManager.close();
      _partitionUpsertMetadataManager = null;
    }
  }

  @Test
  public void testHashFunctions()
      throws Exception {
    testUpsertIngestion(createFullUpsertConfig(HashFunction.NONE));
    testUpsertIngestion(createFullUpsertConfig(HashFunction.MD5));
    testUpsertIngestion(createFullUpsertConfig(HashFunction.MURMUR3));
  }

  /// The upsert metadata manager captures the primary key columns once, when the server starts. A consuming
  /// segment created later picks up whatever schema is current, which may list a different set. Both must key
  /// records identically, otherwise the same row hashes under two different keys and neither invalidates the other.
  @Test
  public void testPrimaryKeyColumnsTakenFromUpsertContext()
      throws Exception {
    TableConfig tableConfig = new TableConfigBuilder(TableType.REALTIME).setTableName(RAW_TABLE_NAME)
        .setTimeColumnName(TIME_COLUMN)
        .setUpsertConfig(createFullUpsertConfig(HashFunction.MURMUR3))
        .setNullHandlingEnabled(true)
        .build();
    URL schemaResourceUrl = getClass().getClassLoader().getResource(SCHEMA_FILE_PATH);
    assertNotNull(schemaResourceUrl);
    // Schema as it was when the metadata manager was created: a single primary key column.
    Schema managerSchema = Schema.fromFile(new File(schemaResourceUrl.getFile()));
    // Schema the consuming segment was created with: a second column has since joined the primary key.
    Schema segmentSchema = Schema.fromFile(new File(schemaResourceUrl.getFile()));
    segmentSchema.setPrimaryKeyColumns(List.of(PRIMARY_KEY_COLUMN, OTHER_COMPARISON_COLUMN));

    TableUpsertMetadataManager tableUpsertMetadataManager =
        TableUpsertMetadataManagerFactory.create(new PinotConfiguration(), tableConfig, managerSchema,
            mock(TableDataManager.class), null);
    PartitionUpsertMetadataManager partitionUpsertMetadataManager =
        tableUpsertMetadataManager.getOrCreatePartitionManager(0);
    MutableSegmentImpl mutableSegment =
        MutableSegmentImplTestUtils.createMutableSegmentImpl(segmentSchema, true, TIME_COLUMN,
            partitionUpsertMetadataManager, null);
    try {
      // Two records sharing event_id, differing only on the column that the segment schema treats as part of the
      // key but the metadata manager does not.
      mutableSegment.index(createRow("aa", 1L, 1567205396L), null);
      mutableSegment.index(createRow("aa", 2L, 1567205397L), null);

      // Keyed on the manager's columns these are the same record, so the older doc must have been invalidated.
      ImmutableRoaringBitmap bitmap = mutableSegment.getValidDocIds().getMutableRoaringBitmap();
      assertEquals(bitmap.getCardinality(), 1);
      assertFalse(bitmap.contains(0));
      assertTrue(bitmap.contains(1));
    } finally {
      mutableSegment.destroy();
      partitionUpsertMetadataManager.stop();
      partitionUpsertMetadataManager.close();
    }
  }

  private static GenericRow createRow(String eventId, long otherComparisonValue, long timeValue) {
    GenericRow row = new GenericRow();
    row.putValue(PRIMARY_KEY_COLUMN, eventId);
    row.putValue("description", "d");
    row.putValue(OTHER_COMPARISON_COLUMN, otherComparisonValue);
    row.putValue(TIME_COLUMN, timeValue);
    return row;
  }

  @Test
  public void testMultipleComparisonColumns()
      throws Exception {
    testUpsertIngestion(createPartialUpsertConfig(HashFunction.NONE));
    testUpsertIngestion(createPartialUpsertConfig(HashFunction.MD5));
    testUpsertIngestion(createPartialUpsertConfig(HashFunction.MURMUR3));
  }

  /// With several comparison columns each event fills one of them. Live ingestion compares the incoming column and
  /// carries the other columns' last values in the upsert metadata. A commit or a restart rebuilds the metadata from
  /// the stored rows, so the stored rows must hold the same comparison values, or the rebuild picks another winner.
  @Test
  public void testMultipleComparisonColumnsKeepWinnerAfterCommitAndRestart()
      throws Exception {
    Schema schema = new Schema.SchemaBuilder().setSchemaName(RAW_TABLE_NAME)
        .addSingleValueDimension(PRIMARY_KEY_COLUMN, DataType.STRING)
        .addSingleValueDimension("a", DataType.LONG)
        .addSingleValueDimension("b", DataType.LONG)
        .addSingleValueDimension("deleted", DataType.BOOLEAN)
        .setPrimaryKeyColumns(List.of(PRIMARY_KEY_COLUMN))
        .build();
    UpsertConfig upsertConfig = new UpsertConfig(UpsertConfig.Mode.FULL);
    upsertConfig.setComparisonColumns(List.of("a", "b"));
    upsertConfig.setDeleteRecordColumn("deleted");
    TableConfig tableConfig = new TableConfigBuilder(TableType.REALTIME).setTableName(RAW_TABLE_NAME)
        .setUpsertConfig(upsertConfig)
        .setNullHandlingEnabled(true)
        .build();
    TransformPipeline transformPipeline = new TransformPipeline(tableConfig, schema);
    File segmentDir = new File(FileUtils.getTempDirectory(), "MutableSegmentImplUpsertTest");
    FileUtils.deleteQuietly(segmentDir);

    PartitionUpsertMetadataManager manager = createPartitionManager(tableConfig, schema);
    MutableSegmentImpl mutableSegment =
        MutableSegmentImplTestUtils.createMutableSegmentImpl(schema, true, null, manager, null);
    ImmutableSegmentImpl committedSegment = null;
    PartitionUpsertMetadataManager restartedManager = null;
    ImmutableSegmentImpl restartedSegment = null;
    try {
      // k1: CREATE a=380, DELETE b=382. The delete wins and must stay.
      indexRow(mutableSegment, transformPipeline, "k1", 380L, null, false);
      indexRow(mutableSegment, transformPipeline, "k1", null, 382L, true);
      // k2: CREATE b=10, DELETE a=20, RECREATE b=30. The recreate wins and must stay.
      indexRow(mutableSegment, transformPipeline, "k2", null, 10L, false);
      indexRow(mutableSegment, transformPipeline, "k2", 20L, null, true);
      indexRow(mutableSegment, transformPipeline, "k2", null, 30L, false);
      // k3: CREATE a=380, UPDATE b=383, late a=379. The update wins; the late event must not.
      indexRow(mutableSegment, transformPipeline, "k3", 380L, null, false);
      indexRow(mutableSegment, transformPipeline, "k3", null, 383L, false);
      indexRow(mutableSegment, transformPipeline, "k3", 379L, null, false);
      assertEquals(mutableSegment.getQueryableDocIds().getMutableRoaringBitmap().toArray(), new int[]{4, 6});

      // Commit: build the immutable segment from the stored rows and replace the consuming segment with it
      List<GenericRow> rows = new ArrayList<>();
      for (int docId = 0; docId < mutableSegment.getNumDocsIndexed(); docId++) {
        rows.add(mutableSegment.getRecord(docId, new GenericRow()));
      }
      SegmentGeneratorConfig generatorConfig = new SegmentGeneratorConfig(tableConfig, schema);
      generatorConfig.setOutDir(segmentDir.getAbsolutePath());
      generatorConfig.setSegmentName(mutableSegment.getSegmentName());
      generatorConfig.setTableName(RAW_TABLE_NAME);
      SegmentIndexCreationDriverImpl driver = new SegmentIndexCreationDriverImpl();
      driver.init(generatorConfig, new GenericRowRecordReader(rows));
      driver.build();
      File indexDir = new File(segmentDir, mutableSegment.getSegmentName());
      committedSegment = (ImmutableSegmentImpl) ImmutableSegmentLoader.load(indexDir, ReadMode.mmap);
      // The first event of k1 carries only column a
      assertTrue(committedSegment.getDataSource("b").getNullValueVector().isNull(0));
      manager.replaceSegment(committedSegment, mutableSegment);
      assertEquals(committedSegment.getQueryableDocIds().getMutableRoaringBitmap().toArray(), new int[]{4, 6});

      // Restart: a new metadata manager rebuilds from the committed segment alone
      restartedManager = createPartitionManager(tableConfig, schema);
      restartedSegment = (ImmutableSegmentImpl) ImmutableSegmentLoader.load(indexDir, ReadMode.mmap);
      restartedManager.addSegment(restartedSegment);
      assertEquals(restartedSegment.getQueryableDocIds().getMutableRoaringBitmap().toArray(), new int[]{4, 6});
    } finally {
      mutableSegment.destroy();
      if (committedSegment != null) {
        committedSegment.destroy();
      }
      if (restartedSegment != null) {
        restartedSegment.destroy();
      }
      manager.stop();
      manager.close();
      if (restartedManager != null) {
        restartedManager.stop();
        restartedManager.close();
      }
      FileUtils.deleteQuietly(segmentDir);
    }
  }

  private static PartitionUpsertMetadataManager createPartitionManager(TableConfig tableConfig, Schema schema) {
    return TableUpsertMetadataManagerFactory.create(new PinotConfiguration(), tableConfig, schema,
        mock(TableDataManager.class), null).getOrCreatePartitionManager(0);
  }

  private static void indexRow(MutableSegmentImpl segment, TransformPipeline transformPipeline, String key,
      Long a, Long b, boolean deleted)
      throws Exception {
    GenericRow row = new GenericRow();
    row.putValue(PRIMARY_KEY_COLUMN, key);
    row.putValue("a", a);
    row.putValue("b", b);
    row.putValue("deleted", deleted);
    for (GenericRow transformedRow : transformPipeline.processRow(row).getTransformedRows()) {
      segment.index(transformedRow, null);
    }
  }

  private void testUpsertIngestion(UpsertConfig upsertConfig)
      throws Exception {
    setup(upsertConfig);
    try {
      ImmutableRoaringBitmap bitmap = _mutableSegmentImpl.getValidDocIds().getMutableRoaringBitmap();
      if (upsertConfig.getComparisonColumns() == null) {
        // aa
        assertFalse(bitmap.contains(0));
        assertTrue(bitmap.contains(1));
        assertFalse(bitmap.contains(2));
        assertFalse(bitmap.contains(3));
        // bb
        assertFalse(bitmap.contains(4));
        assertTrue(bitmap.contains(5));
        assertFalse(bitmap.contains(6));
      } else {
        // aa
        assertFalse(bitmap.contains(0));
        assertFalse(bitmap.contains(1));
        assertTrue(bitmap.contains(2));
        assertFalse(bitmap.contains(3));
        // Confirm that both comparison column values have made it into the persisted upserted doc
        assertEquals(_mutableSegmentImpl.getValue(2, TIME_COLUMN), 1567205397L);
        assertEquals(_mutableSegmentImpl.getValue(2, OTHER_COMPARISON_COLUMN), 1567205395L);
        assertFalse(_mutableSegmentImpl.getDataSource(TIME_COLUMN).getNullValueVector().isNull(2));
        assertFalse(_mutableSegmentImpl.getDataSource(OTHER_COMPARISON_COLUMN).getNullValueVector().isNull(2));

        // bb
        assertFalse(bitmap.contains(4));
        assertTrue(bitmap.contains(5));
        assertFalse(bitmap.contains(6));
        // Confirm that comparison column values have made it into the persisted upserted doc
        assertEquals(_mutableSegmentImpl.getValue(5, TIME_COLUMN), 1567205396L);
        assertEquals(_mutableSegmentImpl.getValue(5, OTHER_COMPARISON_COLUMN), Long.MIN_VALUE);
        assertFalse(_mutableSegmentImpl.getDataSource(TIME_COLUMN).getNullValueVector().isNull(5));
        assertTrue(_mutableSegmentImpl.getDataSource(OTHER_COMPARISON_COLUMN).getNullValueVector().isNull(5));
      }
    } finally {
      tearDown();
    }
  }
}

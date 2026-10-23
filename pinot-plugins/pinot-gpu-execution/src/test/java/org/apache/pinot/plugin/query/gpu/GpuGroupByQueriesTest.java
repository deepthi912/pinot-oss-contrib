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

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.io.FileUtils;
import org.apache.pinot.common.datatable.DataTableFactory;
import org.apache.pinot.common.request.BrokerRequest;
import org.apache.pinot.common.request.PinotQuery;
import org.apache.pinot.common.response.broker.BrokerResponseNative;
import org.apache.pinot.core.plan.Plan;
import org.apache.pinot.core.plan.maker.InstancePlanMakerImplV2;
import org.apache.pinot.core.plan.maker.PlanMaker;
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.core.query.request.context.utils.QueryContextConverterUtils;
import org.apache.pinot.core.transport.ServerRoutingInstance;
import org.apache.pinot.plugin.query.gpu.JavaGpuBackendProvider.Behavior;
import org.apache.pinot.plugin.query.gpu.metal.MetalGpuBackend;
import org.apache.pinot.plugin.query.gpu.metal.MetalGpuBackendProvider;
import org.apache.pinot.plugin.query.gpu.metal.MetalNativeLibrary;
import org.apache.pinot.queries.BaseQueriesTest;
import org.apache.pinot.segment.local.indexsegment.immutable.ImmutableSegmentLoader;
import org.apache.pinot.segment.local.segment.creator.impl.SegmentIndexCreationDriverImpl;
import org.apache.pinot.segment.local.segment.readers.GenericRowRecordReader;
import org.apache.pinot.segment.spi.ImmutableSegment;
import org.apache.pinot.segment.spi.IndexSegment;
import org.apache.pinot.segment.spi.MutableSegment;
import org.apache.pinot.segment.spi.SegmentContext;
import org.apache.pinot.segment.spi.creator.SegmentGeneratorConfig;
import org.apache.pinot.spi.config.table.TableConfig;
import org.apache.pinot.spi.config.table.TableType;
import org.apache.pinot.spi.data.FieldSpec.DataType;
import org.apache.pinot.spi.data.Schema;
import org.apache.pinot.spi.data.readers.GenericRow;
import org.apache.pinot.spi.env.PinotConfiguration;
import org.apache.pinot.spi.query.QueryThreadContext;
import org.apache.pinot.spi.utils.ReadMode;
import org.apache.pinot.spi.utils.builder.TableConfigBuilder;
import org.apache.pinot.sql.parsers.CalciteSqlCompiler;
import org.apache.pinot.sql.parsers.CalciteSqlParser;
import org.roaringbitmap.buffer.MutableRoaringBitmap;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import static org.mockito.AdditionalAnswers.delegatesTo;
import static org.mockito.Mockito.mock;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.fail;


/// Runs every query through the regular CPU plan maker and through [GpuPlanMaker], end to end (segment plan, combine,
/// data table serialization, broker reduce), and requires identical rows and scan statistics. Also asserts whether the
/// GPU backend was called, so eligible queries are proven to take the GPU path and others the CPU path.
///
/// Metal kernels (macOS on Apple Silicon): `./mvnw -pl pinot-plugins/pinot-gpu-execution -am test -Ppinot-fastdev
/// -Dpinot.gpu.metal=true -Dtest=GpuGroupByQueriesTest -Dsurefire.failIfNoSpecifiedTests=false`; with that property
/// [#testMetalBackendEndToEnd] fails instead of skipping when Metal is unavailable.
public class GpuGroupByQueriesTest extends BaseQueriesTest {
  private static final File INDEX_DIR = new File(FileUtils.getTempDirectory(), "GpuGroupByQueriesTest");
  private static final String TABLE = "testTable";
  private static final int NUM_ROWS_PER_SEGMENT = 20_003;
  private static final String METAL_PROPERTY = "pinot.gpu.metal";
  private static final String QUERY =
      "SELECT dimStr, COUNT(*) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000";
  private static final String GROUP_BY_VALUES_QUERY = "SELECT dimStr, SUM(metric), AVG(rawMetric), "
      + "DISTINCTCOUNT(dimInt) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000";
  private static final String AGGREGATION_QUERY =
      "SELECT SUM(metric), AVG(rawMetric), DISTINCTCOUNT(dimStr), COUNT(*) FROM testTable WHERE metric > 100";

  private static final Schema SCHEMA = new Schema.SchemaBuilder().setSchemaName(TABLE)
      .addSingleValueDimension("dimStr", DataType.STRING)
      .addSingleValueDimension("dimInt", DataType.INT)
      .addSingleValueDimension("dimHigh", DataType.INT)
      .addSingleValueDimension("dimLong", DataType.LONG)
      .addSingleValueDimension("dimFloat", DataType.FLOAT)
      .addSingleValueDimension("dimDouble", DataType.DOUBLE)
      .addSingleValueDimension("dimBool", DataType.BOOLEAN)
      .addSingleValueDimension("dimTs", DataType.TIMESTAMP)
      .addSingleValueDimension("dimBytes", DataType.BYTES)
      .addSingleValueDimension("rawStr", DataType.STRING)
      .addMultiValueDimension("mvInt", DataType.INT)
      .addMetric("metric", DataType.INT)
      .addMetric("rawMetric", DataType.INT)
      .build();
  private static final TableConfig TABLE_CONFIG = new TableConfigBuilder(TableType.OFFLINE).setTableName(TABLE)
      .setNoDictionaryColumns(List.of("rawStr", "rawMetric")).build();

  private final JavaGpuBackendProvider _gpuProvider =
      new JavaGpuBackendProvider(GpuBackendType.CUDA, true, Behavior.CORRECT);
  private final JavaGpuBackendProvider _failingProvider =
      new JavaGpuBackendProvider(GpuBackendType.CUDA, true, Behavior.FAIL);
  private final JavaGpuBackendProvider _corruptValuesProvider =
      new JavaGpuBackendProvider(GpuBackendType.CUDA, true, Behavior.CORRUPT_VALUES);
  private final JavaGpuBackendProvider _corruptProvider =
      new JavaGpuBackendProvider(GpuBackendType.CUDA, true, Behavior.CORRUPT);
  private final PlanMaker _cpuPlanMaker = new InstancePlanMakerImplV2();
  private GpuPlanMaker _gpuPlanMaker;
  private GpuPlanMaker _failingPlanMaker;
  private GpuPlanMaker _corruptPlanMaker;
  private GpuPlanMaker _corruptValuesPlanMaker;
  private List<IndexSegment> _indexSegments;

  @Override
  protected String getFilter() {
    return "";
  }

  @Override
  protected IndexSegment getIndexSegment() {
    return _indexSegments.get(0);
  }

  @Override
  protected List<IndexSegment> getIndexSegments() {
    return _indexSegments;
  }

  @BeforeClass
  public void setUp()
      throws Exception {
    FileUtils.deleteQuietly(INDEX_DIR);
    _indexSegments = List.of(buildSegment("segment0", 0), buildSegment("segment1", 1));
    _gpuPlanMaker = gpuPlanMaker(List.of(_gpuProvider));
    _failingPlanMaker = gpuPlanMaker(List.of(_failingProvider));
    _corruptPlanMaker = gpuPlanMaker(List.of(_corruptProvider));
    _corruptValuesPlanMaker = gpuPlanMaker(List.of(_corruptValuesProvider));
  }

  private static GpuPlanMaker gpuPlanMaker(List<GpuBackendProvider> providers) {
    GpuPlanMaker planMaker = providers != null ? new GpuPlanMaker(providers) : new GpuPlanMaker();
    planMaker.init(new PinotConfiguration(
        Map.of(GpuExecutionConfig.ENABLED, "true", GpuExecutionConfig.MIN_ROWS, "0")));
    return planMaker;
  }

  private static ImmutableSegment buildSegment(String name, long seed)
      throws Exception {
    Random random = new Random(seed);
    List<GenericRow> rows = new ArrayList<>(NUM_ROWS_PER_SEGMENT);
    for (int i = 0; i < NUM_ROWS_PER_SEGMENT; i++) {
      GenericRow row = new GenericRow();
      row.putValue("dimStr", "s" + random.nextInt(100));
      row.putValue("dimInt", random.nextInt(1000));
      row.putValue("dimHigh", random.nextInt(5000));
      row.putValue("dimLong", 10_000_000_000L + random.nextInt(50));
      row.putValue("dimFloat", random.nextInt(40) / 4.0f);
      row.putValue("dimDouble", random.nextInt(40) / 8.0);
      row.putValue("dimBool", random.nextBoolean());
      row.putValue("dimTs", 1_700_000_000_000L + random.nextInt(30) * 3_600_000L);
      row.putValue("dimBytes", new byte[]{(byte) random.nextInt(20), 1});
      row.putValue("rawStr", "r" + random.nextInt(50));
      row.putValue("mvInt", new Object[]{random.nextInt(10), random.nextInt(10)});
      row.putValue("metric", random.nextInt(2001) - 1000);
      row.putValue("rawMetric", random.nextInt() / 4);
      rows.add(row);
    }
    SegmentGeneratorConfig config = new SegmentGeneratorConfig(TABLE_CONFIG, SCHEMA);
    config.setTableName(TABLE);
    config.setSegmentName(name);
    config.setOutDir(INDEX_DIR.getPath());
    SegmentIndexCreationDriverImpl driver = new SegmentIndexCreationDriverImpl();
    driver.init(config, new GenericRowRecordReader(rows));
    driver.build();
    return ImmutableSegmentLoader.load(new File(INDEX_DIR, name), ReadMode.mmap);
  }

  @DataProvider
  public Object[][] gpuQueries() {
    return new Object[][]{
        {QUERY},
        {"SELECT dimInt, COUNT(*) FROM testTable GROUP BY dimInt ORDER BY dimInt LIMIT 2000"},
        {"SELECT dimHigh, COUNT(*) FROM testTable GROUP BY dimHigh ORDER BY dimHigh LIMIT 10000"},
        {"SELECT dimLong, COUNT(*) FROM testTable GROUP BY dimLong ORDER BY dimLong LIMIT 100"},
        {"SELECT dimFloat, COUNT(*) FROM testTable GROUP BY dimFloat ORDER BY dimFloat LIMIT 100"},
        {"SELECT dimDouble, COUNT(*) FROM testTable GROUP BY dimDouble ORDER BY dimDouble LIMIT 100"},
        {"SELECT dimBool, COUNT(*) FROM testTable GROUP BY dimBool ORDER BY dimBool LIMIT 10"},
        {"SELECT dimTs, COUNT(*) FROM testTable GROUP BY dimTs ORDER BY dimTs LIMIT 100"},
        {"SELECT dimBytes, COUNT(*) FROM testTable GROUP BY dimBytes ORDER BY dimBytes LIMIT 100"},
        {"SELECT dimStr, COUNT(*) FROM testTable WHERE metric > 100 GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SELECT dimStr, COUNT(*) FROM testTable GROUP BY dimStr ORDER BY COUNT(*) DESC, dimStr LIMIT 5"},
        {"SELECT dimStr, COUNT(*) AS c FROM testTable GROUP BY dimStr HAVING c > 400 ORDER BY dimStr LIMIT 1000"},
        {"SELECT dimStr, COUNT(*), COUNT(*) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SELECT dimStr, COUNT(metric) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SELECT dimInt, COUNT(*) FROM testTable GROUP BY dimInt ORDER BY dimInt LIMIT 7"},
        {"SELECT dimInt, COUNT(*) FROM testTable GROUP BY dimInt ORDER BY dimInt DESC LIMIT 7"},
        {"SELECT dimStr, COUNT(*) FROM testTable WHERE metric > 0 GROUP BY dimStr ORDER BY dimStr DESC LIMIT 3"},
        {"SET numGroupsLimit = 1001; "
            + "SELECT dimInt, COUNT(*) FROM testTable GROUP BY dimInt ORDER BY dimInt LIMIT 2000"},
        {"SET numGroupsWarningLimit = 10; " + QUERY},
        // Set by the broker whenever one server serves the query; final results are produced after the combine.
        {"SET serverReturnFinalResult = true; " + QUERY},
        {"SET serverReturnFinalResult = true; "
            + "SELECT dimStr, COUNT(*) FROM testTable GROUP BY dimStr ORDER BY COUNT(*) DESC, dimStr LIMIT 5"},
        // Segment trim below the number of groups through the sorted combine, both directions.
        {"SET minSegmentGroupTrimSize = 10; "
            + "SELECT dimInt, COUNT(*) FROM testTable GROUP BY dimInt ORDER BY dimInt LIMIT 3"},
        {"SET minSegmentGroupTrimSize = 10; "
            + "SELECT dimInt, SUM(metric) FROM testTable GROUP BY dimInt ORDER BY dimInt DESC LIMIT 3"},
        // SUM / AVG on dictionary and raw INT columns, DISTINCTCOUNT over several types, all mixed.
        {"SELECT dimStr, SUM(metric), AVG(metric) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SELECT dimStr, SUM(rawMetric), AVG(rawMetric) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SELECT dimStr, DISTINCTCOUNT(dimInt) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SELECT dimStr, DISTINCTCOUNT(dimLong), DISTINCTCOUNT(dimDouble), DISTINCTCOUNT(dimBytes) FROM testTable "
            + "GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SELECT dimStr, DISTINCTCOUNT(dimFloat), DISTINCTCOUNT(dimTs) FROM testTable GROUP BY dimStr "
            + "ORDER BY dimStr LIMIT 1000"},
        {"SELECT DISTINCTCOUNT(dimFloat), DISTINCTCOUNT(dimTs) FROM testTable WHERE metric > 0"},
        {"SELECT dimStr, SUM(dimBool), AVG(dimBool) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SELECT dimInt, COUNT(*), SUM(metric), AVG(rawMetric), DISTINCTCOUNT(dimStr) FROM testTable "
            + "GROUP BY dimInt ORDER BY dimInt LIMIT 2000"},
        {"SELECT dimStr, SUM(metric) FROM testTable WHERE metric > 0 GROUP BY dimStr "
            + "ORDER BY SUM(metric) DESC, dimStr LIMIT 5"},
        // More than 1024 groups: Metal's global-memory SUM kernel.
        {"SELECT dimHigh, SUM(rawMetric), AVG(metric) FROM testTable GROUP BY dimHigh ORDER BY dimHigh LIMIT 10000"},
        // Aggregation without GROUP BY; with SUM present core would scan, so COUNT(*) comes from the GPU too.
        {"SELECT SUM(metric), AVG(metric) FROM testTable"},
        {"SELECT COUNT(*), SUM(metric), AVG(rawMetric) FROM testTable"},
        // One column as both SUM/AVG values and DISTINCTCOUNT dictionary ids, in both upload orders.
        {"SELECT dimStr, SUM(dimInt), AVG(dimInt), DISTINCTCOUNT(dimInt) FROM testTable GROUP BY dimStr "
            + "ORDER BY dimStr LIMIT 1000"},
        {"SELECT dimStr, DISTINCTCOUNT(dimInt), SUM(dimInt) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SELECT SUM(metric), DISTINCTCOUNT(metric), COUNT(*) FROM testTable WHERE metric > 100"},
        {"SELECT SUM(rawMetric), AVG(rawMetric), DISTINCTCOUNT(dimStr), COUNT(*) FROM testTable WHERE metric > 100"},
        {"SELECT DISTINCTCOUNT(dimHigh) FROM testTable WHERE dimBool = true"}
    };
  }

  @DataProvider
  public Object[][] cpuQueries() {
    return new Object[][]{
        {"SELECT COUNT(*) FROM testTable"},
        {"SELECT dimStr, SUM(dimLong) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SELECT dimStr, MIN(metric) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SELECT dimStr, DISTINCTCOUNT(rawStr) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        // Without a filter, Pinot answers these from the dictionary or metadata without scanning.
        {"SELECT DISTINCTCOUNT(dimStr) FROM testTable"},
        {"SELECT COUNT(*), DISTINCTCOUNT(dimStr) FROM testTable"},
        {"SELECT dimStr, dimInt, COUNT(*) FROM testTable GROUP BY dimStr, dimInt ORDER BY dimStr, dimInt LIMIT 100"},
        {"SELECT rawStr, COUNT(*) FROM testTable GROUP BY rawStr ORDER BY rawStr LIMIT 1000"},
        {"SELECT mvInt, COUNT(*) FROM testTable GROUP BY mvInt ORDER BY mvInt LIMIT 1000"},
        {"SELECT dimStr FROM testTable ORDER BY dimStr LIMIT 10"},
        {"SELECT dimStr, COUNT(*) FROM testTable GROUP BY dimStr LIMIT 0"},
        {"SELECT dimStr, COUNT(*) FILTER (WHERE metric > 0) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SET enableNullHandling = true; " + QUERY},
        {"SET useGpu = false; " + QUERY},
        {"SET gpuBackend = 'cpu'; " + QUERY},
        // dimInt has 1000 values per segment: the GPU needs fewer than numGroupsLimit; 500 makes the CPU cap groups.
        {"SET numGroupsLimit = 500; "
            + "SELECT dimInt, COUNT(*) FROM testTable GROUP BY dimInt ORDER BY dimInt LIMIT 2000"},
        {"SET numGroupsLimit = 1000; "
            + "SELECT dimInt, COUNT(*) FROM testTable GROUP BY dimInt ORDER BY dimInt LIMIT 2000"},
        {"SELECT dimInt, COUNT(*) FROM testTable WHERE dimStr IN ('s1', 's2', 's3') GROUP BY dimInt "
            + "ORDER BY dimInt LIMIT 2000"},
        {"SELECT dimStr, COUNT(*) FROM testTable WHERE dimInt < 20 GROUP BY dimStr ORDER BY dimStr LIMIT 1000"}
    };
  }

  @Test(dataProvider = "gpuQueries")
  public void testEligibleQueriesRunOnGpu(String query)
      throws Exception {
    assertSameResults(query, _gpuPlanMaker, _gpuProvider, true);
    // Matching results alone would also pass on a CPU fallback.
    GpuQueryStats stats = executeAndGetStats(_gpuPlanMaker, query);
    assertEquals(stats.getSegmentsFallbackToCpu(), 0, stats.toString());
    assertEquals(stats.getSegmentsOnGpu(), _indexSegments.size(), stats.toString());
  }

  @DataProvider
  public Object[][] fallbackQueries() {
    Object[][] valueQueries = valueFallbackQueries();
    Object[][] queries = new Object[valueQueries.length + 1][];
    queries[0] = new Object[]{QUERY};
    System.arraycopy(valueQueries, 0, queries, 1, valueQueries.length);
    return queries;
  }

  /// Queries with SUM/AVG or DISTINCTCOUNT, whose results are validated beyond the counts.
  @DataProvider
  public Object[][] valueFallbackQueries() {
    return new Object[][]{
        {GROUP_BY_VALUES_QUERY}, {AGGREGATION_QUERY},
        // DISTINCTCOUNT without SUM/AVG, so the distinct-result validation is what triggers the fallback.
        {"SELECT dimStr, DISTINCTCOUNT(dimInt) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SELECT DISTINCTCOUNT(dimStr), COUNT(*) FROM testTable WHERE metric > 100"}
    };
  }

  @Test(dataProvider = "cpuQueries")
  public void testOtherQueriesStayOnCpu(String query) {
    assertSameResults(query, _gpuPlanMaker, _gpuProvider, false);
  }

  @Test
  public void testNoMatchingDocuments() {
    for (String query : List.of(
        "SELECT dimStr, COUNT(*) FROM testTable WHERE metric > 5000 GROUP BY dimStr ORDER BY dimStr LIMIT 10",
        "SELECT SUM(metric), AVG(rawMetric), DISTINCTCOUNT(dimStr), COUNT(*) FROM testTable WHERE metric > 5000")) {
      assertSameResponse(getBrokerResponse(query, _gpuPlanMaker), getBrokerResponse(query, _cpuPlanMaker), query);
    }
  }

  @Test(dataProvider = "fallbackQueries")
  public void testDeviceFailureFallsBackToCpu(String query)
      throws Exception {
    assertAllSegmentsFallBack(query, _failingPlanMaker, _failingProvider);
  }

  @Test(dataProvider = "fallbackQueries")
  public void testCorruptDeviceResultsFallBackToCpu(String query)
      throws Exception {
    assertAllSegmentsFallBack(query, _corruptPlanMaker, _corruptProvider);
  }

  @Test(dataProvider = "valueFallbackQueries")
  public void testInconsistentSumsOrDistinctValuesFallBackToCpu(String query)
      throws Exception {
    assertAllSegmentsFallBack(query, _corruptValuesPlanMaker, _corruptValuesProvider);
  }

  private void assertAllSegmentsFallBack(String query, GpuPlanMaker planMaker, JavaGpuBackendProvider provider)
      throws Exception {
    assertSameResults(query, planMaker, provider, true);
    GpuQueryStats stats = executeAndGetStats(planMaker, query);
    assertEquals(stats.getSegmentsOnGpu(), 0, stats.toString());
    assertEquals(stats.getSegmentsFallbackToCpu(), _indexSegments.size(), stats.toString());
  }

  @Test
  public void testExecutionStatsAccountForEverySegment()
      throws Exception {
    GpuQueryStats stats = executeAndGetStats(_gpuPlanMaker, QUERY);
    assertEquals(stats.getSegmentsOnGpu(), _indexSegments.size(), stats.toString());
    assertEquals(stats.getSegmentsFallbackToCpu(), 0, stats.toString());

    for (String query : List.of(
        "SELECT dimStr, COUNT(*) FROM testTable WHERE dimInt < 20 GROUP BY dimStr ORDER BY dimStr LIMIT 1000",
        "SELECT SUM(metric), AVG(rawMetric), DISTINCTCOUNT(dimStr) FROM testTable WHERE dimInt < 20",
        // 1000 x 5000 (group, value) bits per segment: more bitmap words than matching documents.
        "SELECT dimInt, DISTINCTCOUNT(dimHigh) FROM testTable WHERE metric > 100 GROUP BY dimInt "
            + "ORDER BY dimInt LIMIT 2000")) {
      assertSameResponse(getBrokerResponse(query, _gpuPlanMaker), getBrokerResponse(query, _cpuPlanMaker), query);
      GpuQueryStats selective = executeAndGetStats(_gpuPlanMaker, query);
      assertEquals(selective.getSegmentsOnGpu(), 0, selective.toString());
      assertEquals(selective.getSegmentsSelectiveOnCpu(), _indexSegments.size(), selective.toString());
    }
  }

  @Test
  public void testResidentColumnsAreReused() {
    String query = "SELECT dimInt, COUNT(*) FROM testTable WHERE metric < 0 GROUP BY dimInt ORDER BY dimInt LIMIT 10";
    assertSameResults(query, _gpuPlanMaker, _gpuProvider, true);
    int uploads = _gpuProvider.getNumUploads();
    assertSameResults(query, _gpuPlanMaker, _gpuProvider, true);
    assertEquals(_gpuProvider.getNumUploads(), uploads, "second query must reuse the resident columns");
  }

  @Test
  public void testReloadedSegmentDoesNotReuseStaleColumns()
      throws Exception {
    String query = "SELECT dimStr, SUM(metric), DISTINCTCOUNT(dimInt) FROM testTable WHERE metric >= 0 "
        + "GROUP BY dimStr ORDER BY dimStr LIMIT 1000";
    List<IndexSegment> original = _indexSegments;
    ImmutableSegment before = buildSegment("reloaded", 100);
    ImmutableSegment after = null;
    try {
      _indexSegments = List.of(before);
      assertSameResults(query, _gpuPlanMaker, _gpuProvider, true);
      before.destroy();
      // Same name and row count, different data: the cache must not serve the old instance's columns.
      after = buildSegment("reloaded", 101);
      _indexSegments = List.of(after);
      int uploads = _gpuProvider.getNumUploads();
      assertSameResults(query, _gpuPlanMaker, _gpuProvider, true);
      assertTrue(_gpuProvider.getNumUploads() > uploads, "reloaded segment must upload its own columns");
    } finally {
      _indexSegments = original;
      if (after != null) {
        after.destroy();
      }
    }
  }

  @Test
  public void testColumnsThatDoNotFitStayOnCpu()
      throws Exception {
    GpuPlanMaker planMaker = new GpuPlanMaker(List.of(_gpuProvider));
    planMaker.init(new PinotConfiguration(Map.of(GpuExecutionConfig.ENABLED, "true", GpuExecutionConfig.MIN_ROWS,
        "0", GpuExecutionConfig.CACHE_MAX_BYTES, "1")));
    try {
      for (String query : List.of(QUERY, AGGREGATION_QUERY)) {
        assertSameResults(query, planMaker, _gpuProvider, false);
        GpuQueryStats stats = executeAndGetStats(planMaker, query);
        assertEquals(stats.getSegmentsNotResident(), _indexSegments.size(), stats.toString());
        assertEquals(stats.getSegmentsOnGpu(), 0, stats.toString());
      }
    } finally {
      planMaker.close();
    }
  }

  /// Upsert tables pass each segment's queryable documents as a snapshot bitmap on the segment context.
  @Test
  public void testQueryableDocIdsSnapshotMatchesCpu()
      throws Exception {
    Random random = new Random(3);
    List<SegmentContext> segmentContexts = new ArrayList<>();
    for (IndexSegment segment : _indexSegments) {
      MutableRoaringBitmap queryableDocIds = new MutableRoaringBitmap();
      for (int docId = 0; docId < NUM_ROWS_PER_SEGMENT; docId++) {
        if (random.nextBoolean()) {
          queryableDocIds.add(docId);
        }
      }
      SegmentContext segmentContext = new SegmentContext(segment);
      segmentContext.setQueryableDocIdsSnapshot(queryableDocIds);
      segmentContexts.add(segmentContext);
    }
    for (String query : List.of(
        "SELECT COUNT(*), SUM(metric), AVG(rawMetric), DISTINCTCOUNT(dimInt) FROM testTable",
        "SELECT DISTINCTCOUNT(dimStr) FROM testTable",
        "SELECT dimStr, COUNT(*) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000",
        "SELECT dimStr, COUNT(*), SUM(metric), AVG(rawMetric), DISTINCTCOUNT(dimInt) FROM testTable "
            + "GROUP BY dimStr ORDER BY dimStr LIMIT 1000",
        "SELECT dimStr, SUM(metric), DISTINCTCOUNT(dimInt) FROM testTable WHERE metric > 0 GROUP BY dimStr "
            + "ORDER BY dimStr LIMIT 1000")) {
      BrokerResponseNative expected = getBrokerResponse(query, _cpuPlanMaker, segmentContexts);
      int callsBefore = _gpuProvider.getNumBackendCalls();
      BrokerResponseNative actual = getBrokerResponse(query, _gpuPlanMaker, segmentContexts);
      assertSameResponse(actual, expected, query);
      assertTrue(_gpuProvider.getNumBackendCalls() > callsBefore, query);
      assertTrue(expected.getNumDocsScanned() < (long) NUM_ROWS_PER_SEGMENT * _indexSegments.size(), query);
      GpuQueryStats stats = executeAndGetStats(_gpuPlanMaker, query, segmentContexts);
      assertEquals(stats.getSegmentsOnGpu(), _indexSegments.size(), query + " " + stats);
    }
  }

  /// A consuming segment is a mutable [IndexSegment], not an [ImmutableSegment]; this one serves immutable data.
  @Test
  public void testConsumingSegmentStaysOnCpu()
      throws Exception {
    IndexSegment consuming = mock(MutableSegment.class, delegatesTo(_indexSegments.get(0)));
    List<SegmentContext> segmentContexts = List.of(new SegmentContext(consuming));
    for (String query : List.of(GROUP_BY_VALUES_QUERY, AGGREGATION_QUERY)) {
      BrokerResponseNative expected = getBrokerResponse(query, _cpuPlanMaker, segmentContexts);
      int callsBefore = _gpuProvider.getNumBackendCalls();
      assertSameResponse(getBrokerResponse(query, _gpuPlanMaker, segmentContexts), expected, query);
      GpuQueryStats stats = executeAndGetStats(_gpuPlanMaker, query, segmentContexts);
      assertEquals(_gpuProvider.getNumBackendCalls(), callsBefore, query);
      assertEquals(stats.getSegmentsIneligible(), 1, query + " " + stats);
      assertEquals(stats.getSegmentsOnGpu(), 0, query + " " + stats);
    }
  }

  /// Server and broker over the given segment contexts, which may carry document snapshots.
  private BrokerResponseNative getBrokerResponse(String query, PlanMaker planMaker,
      List<SegmentContext> segmentContexts)
      throws Exception {
    PinotQuery pinotQuery = CalciteSqlParser.compileToPinotQuery(query);
    QueryContext queryContext = QueryContextConverterUtils.getQueryContext(pinotQuery);
    queryContext.setEndTimeMs(System.currentTimeMillis() + 60_000);
    byte[] dataTable;
    try (QueryThreadContext ignore = QueryThreadContext.openForSseTest()) {
      dataTable = planMaker.makeInstancePlan(segmentContexts, queryContext, EXECUTOR_SERVICE).execute().toDataTable()
          .toBytes();
    }
    BrokerRequest brokerRequest = CalciteSqlCompiler.convertToBrokerRequest(pinotQuery);
    return reduceOnDataTable(brokerRequest, brokerRequest,
        Map.of(new ServerRoutingInstance("localhost", 1234, TableType.OFFLINE),
            DataTableFactory.getDataTable(dataTable)));
  }

  @Test
  public void testClosedPlanMakerRunsOnCpu() {
    JavaGpuBackendProvider provider = new JavaGpuBackendProvider(GpuBackendType.CUDA, true, Behavior.CORRECT);
    GpuPlanMaker planMaker = gpuPlanMaker(List.of(provider));
    assertSameResults(GROUP_BY_VALUES_QUERY, planMaker, provider, true);
    planMaker.close();
    assertEquals(provider.getNumBackendCloses(), 1);
    assertSameResults(GROUP_BY_VALUES_QUERY, planMaker, provider, false);
  }

  @Test
  public void testDiscoveryWithoutGpuDegradesToCpu() {
    if (new MetalGpuBackendProvider().probe().isAvailable()) {
      throw new SkipException("Metal is available here, so discovery finds a GPU; see testMetalBackendEndToEnd");
    }
    GpuPlanMaker planMaker = gpuPlanMaker(null);
    assertFalse(planMaker.getRouter().decide(GpuQueryOptions.NONE, 1).runsOnGpu());
    for (Object[] query : gpuQueries()) {
      String sql = (String) query[0];
      assertSameResponse(getBrokerResponse(sql, planMaker), getBrokerResponse(sql, _cpuPlanMaker), sql);
    }
  }

  @Test
  public void testMetalBackendEndToEnd()
      throws Exception {
    GpuBackendProvider.Availability availability = new MetalGpuBackendProvider().probe();
    if (!availability.isAvailable()) {
      if (Boolean.getBoolean(METAL_PROPERTY)) {
        fail("-D" + METAL_PROPERTY + "=true but the Metal backend is unavailable: " + availability);
      }
      throw new SkipException("Metal backend not available: " + availability);
    }
    GpuPlanMaker metalPlanMaker = gpuPlanMaker(null);
    try {
      assertEquals(metalPlanMaker.getRouter().decide(GpuQueryOptions.NONE, 1).getBackend().getType(),
          GpuBackendType.METAL);
      for (Object[] query : gpuQueries()) {
        String sql = (String) query[0];
        assertSameResponse(getBrokerResponse(sql, metalPlanMaker), getBrokerResponse(sql, _cpuPlanMaker), sql);
        GpuQueryStats stats = executeAndGetStats(metalPlanMaker, sql);
        assertEquals(stats.getSegmentsFallbackToCpu(), 0, sql + " " + stats);
        assertEquals(stats.getSegmentsOnGpu(), _indexSegments.size(), sql + " " + stats);
      }
      for (String sql : List.of("SELECT dimInt, COUNT(*) FROM testTable GROUP BY dimInt ORDER BY dimInt LIMIT 2000",
          "SELECT dimHigh, COUNT(*) FROM testTable GROUP BY dimHigh ORDER BY dimHigh LIMIT 10000")) {
        GpuQueryStats stats = executeAndGetStats(metalPlanMaker, sql);
        assertEquals(stats.getSegmentsOnGpu(), _indexSegments.size(), stats.toString());
        assertEquals(stats.getSegmentsFallbackToCpu(), 0, stats.toString());
        assertTrue(stats.getGpuKernelNs() > 0, stats.toString());
      }
      checkMetalKernels(metalPlanMaker.getRouter().decide(GpuQueryOptions.NONE, 1).getBackend());
    } finally {
      metalPlanMaker.close();
    }
    checkMetalClose();
  }

  private static void checkMetalKernels(GpuBackend metal) {
    int[] keys = new int[1_000_000];
    for (int i = 0; i < keys.length; i++) {
      keys[i] = i % 7;
    }
    int[] values = new int[keys.length];
    for (int i = 0; i < values.length; i++) {
      values[i] = i % 2 == 0 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
    }
    try (GpuColumn column = metal.uploadIntColumn(keys, keys.length);
        GpuColumn valueColumn = metal.uploadIntColumn(values, values.length)) {
      GpuAggregationResult result = metal.aggregate(column, 7, valueColumn, keys.length, null);
      assertTrue(result.getDeviceTimeNs() > 0, "Metal reported no device time");
      long[] expectedSums = new long[7];
      long[] expectedCounts = new long[7];
      for (int i = 0; i < keys.length; i++) {
        expectedSums[keys[i]] += values[i];
        expectedCounts[keys[i]]++;
      }
      assertEquals(result.getCounts(), expectedCounts);
      assertEquals(result.getSums(), expectedSums);
      assertEquals(metal.distinct(null, 1, column, 7, keys.length, null).getIds(0, 7).getCardinality(), 7);
    }

    // Global-memory SUM kernel: more than kLocalMaxGroups groups, asymmetric full-range values, per-group sums beyond
    // int32 and more than kGlobalBatchRows rows (several fold batches); constants of metal_ops.mm.
    int numRows = (1 << 24) + 12_345;
    int numGroups = 3001;
    int[] groupKeys = new int[numRows];
    int[] bigValues = new int[numRows];
    long[] expectedGroupSums = new long[numGroups];
    long[] expectedGroupCounts = new long[numGroups];
    Random random = new Random(7);
    for (int i = 0; i < numRows; i++) {
      groupKeys[i] = random.nextInt(numGroups);
      bigValues[i] = random.nextInt() / (groupKeys[i] % 3 == 0 ? 1 : 7) + (groupKeys[i] % 2 == 0 ? 1_000_000 : 0);
      expectedGroupSums[groupKeys[i]] += bigValues[i];
      expectedGroupCounts[groupKeys[i]]++;
    }
    try (GpuColumn keyColumn = metal.uploadIntColumn(groupKeys, numRows);
        GpuColumn valueColumn = metal.uploadIntColumn(bigValues, numRows)) {
      GpuAggregationResult result = metal.aggregate(keyColumn, numGroups, valueColumn, numRows, null);
      assertEquals(result.getCounts(), expectedGroupCounts);
      assertEquals(result.getSums(), expectedGroupSums);
    }

    // Filtered kernels without group keys, and grouped distinct pairs: every 3rd document matches.
    int numDocs = 300_001;
    int[] filterWords = new int[(numDocs + 31) >>> 5];
    long expectedCount = 0;
    long expectedSum = 0;
    int[] groups = new int[numDocs];
    int[] ids = new int[numDocs];
    boolean[][] expectedPairs = new boolean[5][11];
    for (int docId = 0; docId < numDocs; docId++) {
      groups[docId] = docId % 5;
      ids[docId] = (docId / 5) % 11;
      if (docId % 3 == 0) {
        filterWords[docId >>> 5] |= 1 << docId;
        expectedCount++;
        expectedSum += docId - 150_000;
        expectedPairs[groups[docId]][ids[docId]] = true;
      }
    }
    int[] signedValues = new int[numDocs];
    for (int docId = 0; docId < numDocs; docId++) {
      signedValues[docId] = docId - 150_000;
    }
    try (GpuColumn filter = metal.uploadFilter(filterWords);
        GpuColumn valueColumn = metal.uploadIntColumn(signedValues, numDocs);
        GpuColumn groupColumn = metal.uploadIntColumn(groups, numDocs);
        GpuColumn idColumn = metal.uploadIntColumn(ids, numDocs)) {
      GpuAggregationResult total = metal.aggregate(null, 1, valueColumn, numDocs, filter);
      assertEquals(total.getCounts(), new long[]{expectedCount});
      assertEquals(total.getSums(), new long[]{expectedSum});
      GpuDistinctResult pairs = metal.distinct(groupColumn, 5, idColumn, 11, numDocs, filter);
      for (int g = 0; g < 5; g++) {
        for (int id = 0; id < 11; id++) {
          assertEquals(pairs.getIds(g, 11).contains(id), expectedPairs[g][id], "group " + g + " id " + id);
        }
      }
    }
  }

  /// After close, calls are rejected (the plan falls back to the CPU) and a column still leased by a query can be
  /// closed without calling into the unloaded library.
  private static void checkMetalClose() {
    MetalGpuBackend metal = new MetalGpuBackend(MetalNativeLibrary.resolve());
    GpuColumn column = metal.uploadIntColumn(new int[]{1, 2, 3}, 3);
    try {
      assertEquals(metal.aggregate(null, 1, column, 3, null).getSums(), new long[]{6});
    } finally {
      metal.close();
    }
    metal.close();
    assertThrows(IllegalStateException.class, () -> metal.aggregate(null, 1, column, 3, null));
    assertThrows(IllegalStateException.class, () -> metal.uploadIntColumn(new int[]{1}, 1));
    column.close();
    column.close();
  }

  private GpuQueryStats executeAndGetStats(GpuPlanMaker planMaker, String query)
      throws Exception {
    List<SegmentContext> segmentContexts = new ArrayList<>();
    for (IndexSegment segment : _indexSegments) {
      segmentContexts.add(new SegmentContext(segment));
    }
    return executeAndGetStats(planMaker, query, segmentContexts);
  }

  private static GpuQueryStats executeAndGetStats(GpuPlanMaker planMaker, String query,
      List<SegmentContext> segmentContexts)
      throws Exception {
    QueryContext queryContext = QueryContextConverterUtils.getQueryContext(query);
    queryContext.setEndTimeMs(System.currentTimeMillis() + 60_000);
    try (QueryThreadContext ignore = QueryThreadContext.openForSseTest()) {
      Plan plan = planMaker.makeInstancePlan(segmentContexts, queryContext, EXECUTOR_SERVICE);
      assertTrue(plan instanceof GpuInstancePlan, "query was not routed to a GPU backend");
      plan.execute();
      return ((GpuInstancePlan) plan).getStats();
    }
  }

  private void assertSameResults(String query, PlanMaker gpuPlanMaker, JavaGpuBackendProvider provider,
      boolean expectGpu) {
    int callsBefore = provider.getNumBackendCalls();
    BrokerResponseNative expected = getBrokerResponse(query, _cpuPlanMaker);
    BrokerResponseNative actual = getBrokerResponse(query, gpuPlanMaker);
    assertSameResponse(actual, expected, query);
    int calls = provider.getNumBackendCalls() - callsBefore;
    if (expectGpu) {
      assertTrue(calls > 0, "expected the GPU path for: " + query);
    } else {
      assertEquals(calls, 0, "expected the CPU path for: " + query);
    }
  }

  private static void assertSameResponse(BrokerResponseNative actual, BrokerResponseNative expected, String query) {
    assertTrue(expected.getExceptions().isEmpty(), "CPU failed: " + expected.getExceptions());
    assertTrue(actual.getExceptions().isEmpty(), "GPU failed: " + actual.getExceptions());
    assertEquals(actual.getResultTable().getDataSchema(), expected.getResultTable().getDataSchema(), query);
    List<Object[]> actualRows = actual.getResultTable().getRows();
    List<Object[]> expectedRows = expected.getResultTable().getRows();
    assertEquals(actualRows.size(), expectedRows.size(), query);
    for (int i = 0; i < expectedRows.size(); i++) {
      assertEquals(actualRows.get(i), expectedRows.get(i), query + " row " + i);
    }
    assertEquals(actual.getNumDocsScanned(), expected.getNumDocsScanned(), query);
    assertEquals(actual.getNumEntriesScannedInFilter(), expected.getNumEntriesScannedInFilter(), query);
    assertEquals(actual.getNumEntriesScannedPostFilter(), expected.getNumEntriesScannedPostFilter(), query);
    assertEquals(actual.getNumSegmentsMatched(), expected.getNumSegmentsMatched(), query);
    assertEquals(actual.isNumGroupsLimitReached(), expected.isNumGroupsLimitReached(), query);
    assertEquals(actual.isNumGroupsWarningLimitReached(), expected.isNumGroupsWarningLimitReached(), query);
    assertEquals(actual.getTotalDocs(), expected.getTotalDocs(), query);
  }

  @AfterClass
  public void tearDown() {
    for (GpuPlanMaker planMaker : List.of(_gpuPlanMaker, _failingPlanMaker, _corruptPlanMaker,
        _corruptValuesPlanMaker)) {
      planMaker.close();
    }
    for (IndexSegment segment : _indexSegments) {
      segment.destroy();
    }
    FileUtils.deleteQuietly(INDEX_DIR);
  }
}

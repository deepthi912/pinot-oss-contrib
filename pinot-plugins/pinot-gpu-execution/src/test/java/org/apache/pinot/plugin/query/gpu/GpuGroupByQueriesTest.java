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
import org.apache.pinot.common.response.broker.BrokerResponseNative;
import org.apache.pinot.core.plan.Plan;
import org.apache.pinot.core.plan.maker.InstancePlanMakerImplV2;
import org.apache.pinot.core.plan.maker.PlanMaker;
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.core.query.request.context.utils.QueryContextConverterUtils;
import org.apache.pinot.plugin.query.gpu.JavaGpuBackendProvider.Behavior;
import org.apache.pinot.plugin.query.gpu.metal.MetalGpuBackendProvider;
import org.apache.pinot.queries.BaseQueriesTest;
import org.apache.pinot.segment.local.indexsegment.immutable.ImmutableSegmentLoader;
import org.apache.pinot.segment.local.segment.creator.impl.SegmentIndexCreationDriverImpl;
import org.apache.pinot.segment.local.segment.readers.GenericRowRecordReader;
import org.apache.pinot.segment.spi.ImmutableSegment;
import org.apache.pinot.segment.spi.IndexSegment;
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
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;


/// Runs every query through the regular CPU plan maker and through [GpuPlanMaker], end to end (segment plan, combine,
/// data table serialization, broker reduce), and requires identical rows and scan statistics. Also asserts whether the
/// GPU backend was called, so eligible queries are proven to take the GPU path and others the CPU path.
public class GpuGroupByQueriesTest extends BaseQueriesTest {
  private static final File INDEX_DIR = new File(FileUtils.getTempDirectory(), "GpuGroupByQueriesTest");
  private static final String TABLE = "testTable";
  private static final int NUM_ROWS_PER_SEGMENT = 20_003;
  private static final String QUERY =
      "SELECT dimStr, COUNT(*) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000";

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
      .build();
  private static final TableConfig TABLE_CONFIG = new TableConfigBuilder(TableType.OFFLINE).setTableName(TABLE)
      .setNoDictionaryColumns(List.of("rawStr")).build();

  private final JavaGpuBackendProvider _gpuProvider =
      new JavaGpuBackendProvider(GpuBackendType.CUDA, true, Behavior.CORRECT);
  private final JavaGpuBackendProvider _failingProvider =
      new JavaGpuBackendProvider(GpuBackendType.CUDA, true, Behavior.FAIL);
  private final JavaGpuBackendProvider _corruptProvider =
      new JavaGpuBackendProvider(GpuBackendType.CUDA, true, Behavior.CORRUPT);
  private final PlanMaker _cpuPlanMaker = new InstancePlanMakerImplV2();
  private GpuPlanMaker _gpuPlanMaker;
  private GpuPlanMaker _failingPlanMaker;
  private GpuPlanMaker _corruptPlanMaker;
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
            + "SELECT dimStr, COUNT(*) FROM testTable GROUP BY dimStr ORDER BY COUNT(*) DESC, dimStr LIMIT 5"}
    };
  }

  @DataProvider
  public Object[][] cpuQueries() {
    return new Object[][]{
        {"SELECT COUNT(*) FROM testTable"},
        {"SELECT dimStr, SUM(metric) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SELECT dimStr, dimInt, COUNT(*) FROM testTable GROUP BY dimStr, dimInt ORDER BY dimStr, dimInt LIMIT 100"},
        {"SELECT rawStr, COUNT(*) FROM testTable GROUP BY rawStr ORDER BY rawStr LIMIT 1000"},
        {"SELECT mvInt, COUNT(*) FROM testTable GROUP BY mvInt ORDER BY mvInt LIMIT 1000"},
        {"SELECT dimStr FROM testTable ORDER BY dimStr LIMIT 10"},
        {"SELECT dimStr, COUNT(*) FROM testTable GROUP BY dimStr LIMIT 0"},
        {"SELECT dimStr, COUNT(*) FILTER (WHERE metric > 0) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 1000"},
        {"SET enableNullHandling = true; " + QUERY},
        {"SET useGpu = false; " + QUERY},
        {"SET gpuBackend = 'cpu'; " + QUERY},
        // dimInt has 1000 values per segment: at or below the groups limit the CPU group-by caps groups.
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
  public void testEligibleQueriesRunOnGpu(String query) {
    assertSameResults(query, _gpuPlanMaker, _gpuProvider, true);
  }

  @Test(dataProvider = "cpuQueries")
  public void testOtherQueriesStayOnCpu(String query) {
    assertSameResults(query, _gpuPlanMaker, _gpuProvider, false);
  }

  @Test
  public void testNoMatchingDocuments() {
    String query =
        "SELECT dimStr, COUNT(*) FROM testTable WHERE metric > 5000 GROUP BY dimStr ORDER BY dimStr LIMIT 10";
    assertSameResponse(getBrokerResponse(query, _gpuPlanMaker), getBrokerResponse(query, _cpuPlanMaker), query);
  }

  @Test
  public void testDeviceFailureFallsBackToCpu()
      throws Exception {
    assertSameResults(QUERY, _failingPlanMaker, _failingProvider, true);
    GpuQueryStats stats = executeAndGetStats(_failingPlanMaker, QUERY);
    assertEquals(stats.getSegmentsOnGpu(), 0, stats.toString());
    assertEquals(stats.getSegmentsFallback(), _indexSegments.size(), stats.toString());
  }

  @Test
  public void testCorruptDeviceResultsFallBackToCpu()
      throws Exception {
    assertSameResults(QUERY, _corruptPlanMaker, _corruptProvider, true);
    GpuQueryStats stats = executeAndGetStats(_corruptPlanMaker, QUERY);
    assertEquals(stats.getSegmentsOnGpu(), 0, stats.toString());
    assertEquals(stats.getSegmentsFallback(), _indexSegments.size(), stats.toString());
  }

  @Test
  public void testExecutionStatsAccountForEverySegment()
      throws Exception {
    GpuQueryStats stats = executeAndGetStats(_gpuPlanMaker, QUERY);
    assertEquals(stats.getSegmentsOnGpu(), _indexSegments.size(), stats.toString());
    assertEquals(stats.getSegmentsFallback(), 0, stats.toString());

    GpuQueryStats selective = executeAndGetStats(_gpuPlanMaker,
        "SELECT dimStr, COUNT(*) FROM testTable WHERE dimInt < 20 GROUP BY dimStr ORDER BY dimStr LIMIT 1000");
    assertEquals(selective.getSegmentsOnGpu(), 0, selective.toString());
    assertEquals(selective.getSegmentsSelectiveOnCpu(), _indexSegments.size(), selective.toString());
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
  public void testDiscoveryWithoutGpuDegradesToCpu() {
    if (new MetalGpuBackendProvider().probe().isAvailable()) {
      throw new SkipException("A GPU backend is available here; covered by testMetalBackendEndToEnd");
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
      throw new SkipException("Metal backend not available: " + availability);
    }
    GpuPlanMaker metalPlanMaker = gpuPlanMaker(null);
    assertEquals(metalPlanMaker.getRouter().decide(GpuQueryOptions.NONE, 1).getBackend().getType(),
        GpuBackendType.METAL);
    for (Object[] query : gpuQueries()) {
      String sql = (String) query[0];
      assertSameResponse(getBrokerResponse(sql, metalPlanMaker), getBrokerResponse(sql, _cpuPlanMaker), sql);
    }

    for (String sql : List.of("SELECT dimInt, COUNT(*) FROM testTable GROUP BY dimInt ORDER BY dimInt LIMIT 2000",
        "SELECT dimHigh, COUNT(*) FROM testTable GROUP BY dimHigh ORDER BY dimHigh LIMIT 10000")) {
      GpuQueryStats stats = executeAndGetStats(metalPlanMaker, sql);
      assertEquals(stats.getSegmentsOnGpu(), _indexSegments.size(), stats.toString());
      assertEquals(stats.getSegmentsFallback(), 0, stats.toString());
      assertTrue(stats.getDeviceTimeNs() > 0, stats.toString());
    }

    GpuBackend metal = metalPlanMaker.getRouter().decide(GpuQueryOptions.NONE, 1).getBackend();
    int[] keys = new int[1_000_000];
    for (int i = 0; i < keys.length; i++) {
      keys[i] = i % 7;
    }
    try (GpuColumn column = metal.uploadIntColumn(keys, keys.length)) {
      GpuGroupByCountResult result = metal.groupByCount(column, 7, null);
      assertTrue(result.getDeviceTimeNs() > 0, "Metal reported no device time");
      assertEquals(result.getCounts()[0], 142858L);
      assertEquals(result.getCounts()[6], 142857L);
    }
    metalPlanMaker.getColumnCache().close();
    metalPlanMaker.getRouter().close();
  }

  private GpuQueryStats executeAndGetStats(GpuPlanMaker planMaker, String query)
      throws Exception {
    QueryContext queryContext = QueryContextConverterUtils.getQueryContext(query);
    queryContext.setEndTimeMs(System.currentTimeMillis() + 60_000);
    List<SegmentContext> segmentContexts = new ArrayList<>();
    for (IndexSegment segment : _indexSegments) {
      segmentContexts.add(new SegmentContext(segment));
    }
    try (QueryThreadContext ignore = QueryThreadContext.openForSseTest()) {
      Plan plan = planMaker.makeInstancePlan(segmentContexts, queryContext, EXECUTOR_SERVICE);
      assertTrue(plan instanceof GpuInstancePlan, "query was not routed to a GPU backend");
      plan.execute();
      return ((GpuInstancePlan) plan).getStats();
    }
  }

  private void assertSameResults(String query, PlanMaker gpuPlanMaker, JavaGpuBackendProvider provider,
      boolean expectGpu) {
    int callsBefore = provider.getNumGroupByCalls();
    BrokerResponseNative expected = getBrokerResponse(query, _cpuPlanMaker);
    BrokerResponseNative actual = getBrokerResponse(query, gpuPlanMaker);
    assertSameResponse(actual, expected, query);
    int calls = provider.getNumGroupByCalls() - callsBefore;
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
    for (GpuPlanMaker planMaker : List.of(_gpuPlanMaker, _failingPlanMaker, _corruptPlanMaker)) {
      planMaker.getColumnCache().close();
      planMaker.getRouter().close();
    }
    for (IndexSegment segment : _indexSegments) {
      segment.destroy();
    }
    FileUtils.deleteQuietly(INDEX_DIR);
  }
}

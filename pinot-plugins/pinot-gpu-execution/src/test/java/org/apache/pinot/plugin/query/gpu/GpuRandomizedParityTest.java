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
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import javax.annotation.Nullable;
import org.apache.commons.io.FileUtils;
import org.apache.pinot.common.datatable.DataTable;
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
import org.apache.pinot.sql.parsers.CalciteSqlCompiler;
import org.apache.pinot.sql.parsers.CalciteSqlParser;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.fail;


/// Randomized differential test: generates seeded random aggregation queries (GROUP BY on 0 to 3 dictionary columns,
/// COUNT / SUM / AVG / DISTINCTCOUNT, filters of varying selectivity on indexed and scanned columns, ORDER BY on keys
/// or aggregations, HAVING, LIMIT and trim / group-limit query options), plus shapes the GPU does not support, and runs
/// each through the CPU plan maker and the [GpuPlanMaker] (Java test backend, and Metal with
/// `-Dpinot.gpu.metal=true`), and through failing and corrupting test backends, where every segment the GPU would run
/// falls back to the CPU. Rows and scan statistics must be identical; routing must never change results.
///
/// `-Dpinot.gpu.fuzz.seed=<long>` and `-Dpinot.gpu.fuzz.queries=<count>` override the default seed and query count. A
/// failure reports the seed, the query and the first differing row, so it reproduces with the same seed.
public class GpuRandomizedParityTest extends BaseQueriesTest {
  private static final File INDEX_DIR = new File(FileUtils.getTempDirectory(), "GpuRandomizedParityTest");
  private static final String TABLE = "testTable";
  private static final String SEED_PROPERTY = "pinot.gpu.fuzz.seed";
  private static final String QUERIES_PROPERTY = "pinot.gpu.fuzz.queries";
  private static final long DEFAULT_SEED = 20261103L;
  private static final int DEFAULT_NUM_QUERIES = 300;
  private static final int[] SEGMENT_SIZES = {7_001, 20_003, 1};
  private static final int MAX_REPORTED_MISMATCHES = 10;

  private static final Schema SCHEMA = new Schema.SchemaBuilder().setSchemaName(TABLE)
      .addSingleValueDimension("dInt", DataType.INT)
      .addSingleValueDimension("dIntMed", DataType.INT)
      .addSingleValueDimension("dLong", DataType.LONG)
      .addSingleValueDimension("dStr", DataType.STRING)
      .addSingleValueDimension("dStrMed", DataType.STRING)
      .addSingleValueDimension("dFloat", DataType.FLOAT)
      .addSingleValueDimension("dDouble", DataType.DOUBLE)
      .addSingleValueDimension("dBool", DataType.BOOLEAN)
      .addSingleValueDimension("dBytes", DataType.BYTES)
      .addSingleValueDimension("dConst", DataType.STRING)
      .addSingleValueDimension("sortedCol", DataType.INT)
      .addSingleValueDimension("invCol", DataType.INT)
      .addMetric("metric", DataType.INT)
      .addMetric("rawMetric", DataType.INT)
      .build();
  private static final TableConfig TABLE_CONFIG = new TableConfigBuilder(TableType.OFFLINE).setTableName(TABLE)
      .setNoDictionaryColumns(List.of("rawMetric")).setSortedColumn("sortedCol")
      .setInvertedIndexColumns(List.of("invCol", "dStr")).build();

  // Dictionary columns: GROUP BY keys and DISTINCTCOUNT inputs the GPU supports.
  private static final List<String> DICTIONARY_COLUMNS =
      List.of("dInt", "dIntMed", "dLong", "dStr", "dStrMed", "dFloat", "dDouble", "dBool", "dBytes", "dConst",
          "sortedCol", "invCol", "metric");
  // Columns with stored type INT: SUM / AVG inputs the GPU supports.
  private static final List<String> INT_COLUMNS =
      List.of("metric", "rawMetric", "dInt", "dIntMed", "sortedCol", "invCol", "dBool");
  private static final List<String> UNSUPPORTED_AGGREGATIONS =
      List.of("MIN(metric)", "MAX(rawMetric)", "SUM(dLong)", "AVG(dDouble)", "DISTINCTCOUNT(rawMetric)",
          "SUM(metric + 1)", "DISTINCTCOUNTHLL(dStr)");
  private static final List<String> UNSUPPORTED_KEYS = List.of("rawMetric", "dInt + 1");
  private static final int[] LIMITS = {1, 3, 10, 50, 1000, 100_000};

  private final PlanMaker _cpuPlanMaker = new InstancePlanMakerImplV2();
  private long _seed;
  private int _numQueries;
  private GpuPlanMaker _javaPlanMaker;
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
    _seed = Long.getLong(SEED_PROPERTY, DEFAULT_SEED);
    _numQueries = Integer.getInteger(QUERIES_PROPERTY, DEFAULT_NUM_QUERIES);
    FileUtils.deleteQuietly(INDEX_DIR);
    Random random = new Random(_seed);
    List<IndexSegment> segments = new ArrayList<>();
    for (int i = 0; i < SEGMENT_SIZES.length; i++) {
      segments.add(buildSegment("segment" + i, SEGMENT_SIZES[i], random.nextLong()));
    }
    _indexSegments = segments;
    _javaPlanMaker = javaPlanMaker(Behavior.CORRECT);
    _failingPlanMaker = javaPlanMaker(Behavior.FAIL);
    _corruptPlanMaker = javaPlanMaker(Behavior.CORRUPT);
  }

  private static GpuPlanMaker javaPlanMaker(Behavior behavior) {
    return gpuPlanMaker(List.of(new JavaGpuBackendProvider(GpuBackendType.CUDA, true, behavior)));
  }

  private static GpuPlanMaker gpuPlanMaker(List<GpuBackendProvider> providers) {
    GpuPlanMaker planMaker = new GpuPlanMaker(providers);
    planMaker.init(new PinotConfiguration(
        Map.of(GpuExecutionConfig.ENABLED, "true", GpuExecutionConfig.MIN_ROWS, "0")));
    return planMaker;
  }

  private static ImmutableSegment buildSegment(String name, int numRows, long seed)
      throws Exception {
    Random random = new Random(seed);
    List<GenericRow> rows = new ArrayList<>(numRows);
    for (int i = 0; i < numRows; i++) {
      GenericRow row = new GenericRow();
      row.putValue("dInt", random.nextInt(50) - 25);
      row.putValue("dIntMed", random.nextInt(1500));
      row.putValue("dLong", (random.nextBoolean() ? 5_000_000_000L : -5_000_000_000L) + random.nextInt(30));
      row.putValue("dStr", "s" + random.nextInt(20));
      row.putValue("dStrMed", "m" + random.nextInt(400));
      row.putValue("dFloat", random.nextInt(12) / 4.0f - 1);
      row.putValue("dDouble", random.nextInt(25) * 0.5 - 3);
      row.putValue("dBool", random.nextBoolean());
      row.putValue("dBytes", new byte[]{(byte) random.nextInt(10), (byte) 0xA5});
      row.putValue("dConst", "k");
      row.putValue("sortedCol", (int) ((long) i * 100 / numRows));
      row.putValue("invCol", random.nextInt(100));
      int bucket = random.nextInt(100);
      int metric;
      if (bucket < 4) {
        metric = Integer.MAX_VALUE - random.nextInt(100);
      } else if (bucket < 8) {
        metric = Integer.MIN_VALUE + random.nextInt(100);
      } else {
        metric = random.nextInt(2001) - 1000;
      }
      row.putValue("metric", metric);
      row.putValue("rawMetric", random.nextInt());
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

  @Test
  public void testJavaBackendMatchesCpu() {
    runAll(_javaPlanMaker, "java", true);
  }

  @Test
  public void testFailingBackendFallbackMatchesCpu() {
    runAll(_failingPlanMaker, "java-fail", false);
  }

  @Test
  public void testCorruptBackendFallbackMatchesCpu() {
    runAll(_corruptPlanMaker, "java-corrupt", false);
  }

  @Test
  public void testMetalBackendMatchesCpu() {
    GpuPlanMaker metalPlanMaker = MetalTestSupport.metalPlanMaker();
    try {
      runAll(metalPlanMaker, "metal", true);
    } finally {
      metalPlanMaker.close();
    }
  }

  /// With `onGpu`, a good share of the queries must run on the GPU; otherwise none may, and some must fall back.
  private void runAll(GpuPlanMaker gpuPlanMaker, String backend, boolean onGpu) {
    Random random = new Random(_seed);
    List<String> mismatches = new ArrayList<>();
    int queriesOnGpu = 0;
    int segmentsOnGpu = 0;
    int segmentsFallback = 0;
    int cpuErrors = 0;
    for (int i = 0; i < _numQueries; i++) {
      GeneratedQuery query = randomQuery(random);
      Execution expected = execute(query._sql, _cpuPlanMaker);
      Execution actual = execute(query._sql, gpuPlanMaker);
      if (actual._stats != null) {
        segmentsOnGpu += actual._stats.getSegmentsOnGpu();
        segmentsFallback += actual._stats.getSegmentsFallbackToCpu();
        if (actual._stats.getSegmentsOnGpu() > 0) {
          queriesOnGpu++;
        }
      }
      if (!expected._response.getExceptions().isEmpty()) {
        cpuErrors++;
      }
      String difference = compare(actual._response, expected._response, query._ordered);
      if (difference != null) {
        mismatches.add(String.format("seed=%d backend=%s query#%d: %s%n  %s%n  gpuStats=%s", _seed, backend, i,
            query._sql, difference, actual._stats));
        if (mismatches.size() >= MAX_REPORTED_MISMATCHES) {
          break;
        }
      }
    }
    System.out.printf("GpuRandomizedParityTest seed=%d backend=%s queries=%d onGpu=%d segmentsOnGpu=%d "
            + "segmentsFallbackToCpu=%d cpuErrors=%d mismatches=%d%n", _seed, backend, _numQueries, queriesOnGpu,
        segmentsOnGpu, segmentsFallback, cpuErrors, mismatches.size());
    if (!mismatches.isEmpty()) {
      fail(mismatches.size() + " CPU/GPU mismatch(es):\n" + String.join("\n", mismatches));
    }
    assertTrue(cpuErrors * 20 < _numQueries, "too many generated queries fail on the CPU: " + cpuErrors);
    if (onGpu) {
      // Keeps the generator honest: a good share of the queries must actually exercise the GPU path.
      assertTrue(queriesOnGpu * 5 >= _numQueries, "too few queries ran on the GPU: " + queriesOnGpu);
    } else {
      assertEquals(segmentsOnGpu, 0);
      assertTrue(segmentsFallback * 5 >= _numQueries, "too few segments fell back: " + segmentsFallback);
    }
  }

  private static final class GeneratedQuery {
    final String _sql;
    // Whether the result rows have a deterministic order; otherwise they are compared as a multiset.
    final boolean _ordered;

    GeneratedQuery(String sql, boolean ordered) {
      _sql = sql;
      _ordered = ordered;
    }
  }

  private static final class Execution {
    final BrokerResponseNative _response;
    @Nullable
    final GpuQueryStats _stats;

    Execution(BrokerResponseNative response, @Nullable GpuQueryStats stats) {
      _response = response;
      _stats = stats;
    }
  }

  private GeneratedQuery randomQuery(Random random) {
    StringBuilder options = new StringBuilder();
    if (random.nextInt(100) < 15) {
      options.append("SET minSegmentGroupTrimSize = ").append(pick(random, 1, 5, 10, 50)).append("; ");
    }
    if (random.nextInt(100) < 15) {
      options.append("SET numGroupsLimit = ").append(pick(random, 1, 10, 100, 1000, 5000, 200_000)).append("; ");
    }
    if (random.nextInt(100) < 5) {
      options.append("SET numGroupsWarningLimit = ").append(pick(random, 1, 10, 1000)).append("; ");
    }
    if (random.nextInt(100) < 5) {
      options.append("SET serverReturnFinalResult = true; ");
    }
    if (random.nextInt(100) < 3) {
      options.append("SET enableNullHandling = true; ");
    }
    if (random.nextInt(100) < 3) {
      options.append("SET useGpu = false; ");
    }
    String where = randomFilter(random);
    if (random.nextInt(100) < 8) {
      return new GeneratedQuery(options + unsupportedShape(random, where), true);
    }

    List<String> keys = new ArrayList<>();
    int numKeys = random.nextInt(4);
    while (keys.size() < numKeys) {
      String key = random.nextInt(100) < 4 ? pick(random, UNSUPPORTED_KEYS) : pick(random, DICTIONARY_COLUMNS);
      if (!keys.contains(key)) {
        keys.add(key);
      }
    }
    Set<String> aggregations = new LinkedHashSet<>();
    int numAggregations = 1 + random.nextInt(3);
    while (aggregations.size() < numAggregations) {
      aggregations.add(randomAggregation(random));
    }

    StringBuilder sql = new StringBuilder(options).append("SELECT ");
    List<String> selectList = new ArrayList<>(keys);
    selectList.addAll(aggregations);
    sql.append(String.join(", ", selectList)).append(" FROM testTable").append(where);
    if (keys.isEmpty()) {
      return new GeneratedQuery(sql.toString(), true);
    }
    sql.append(" GROUP BY ").append(String.join(", ", keys));
    if (random.nextInt(100) < 20) {
      List<String> havingCandidates = new ArrayList<>(aggregations);
      havingCandidates.add("COUNT(*)");
      sql.append(" HAVING ").append(pick(random, havingCandidates)).append(' ').append(pick(random, ">", "<", ">="))
          .append(' ').append(pick(random, -1000, 0, 1, 2, 10, 100, 1000));
    }
    int orderMode = random.nextInt(100);
    if (orderMode < 12) {
      // No ORDER BY: groups come back in no particular order, so fetch them all and compare as a multiset.
      sql.append(" LIMIT 200000");
      return new GeneratedQuery(sql.toString(), false);
    }
    List<String> orderBy = new ArrayList<>();
    if (orderMode < 55) {
      // The GROUP BY columns in order, one direction: the order the GPU sort-aggregate supports.
      String direction = random.nextBoolean() ? " ASC" : " DESC";
      for (String key : keys) {
        orderBy.add(key + direction);
      }
    } else if (orderMode < 75) {
      // The GROUP BY columns permuted, with mixed directions.
      List<String> permuted = new ArrayList<>(keys);
      Collections.shuffle(permuted, random);
      for (String key : permuted) {
        orderBy.add(key + (random.nextBoolean() ? " ASC" : " DESC"));
      }
    } else {
      // An aggregation first, then every GROUP BY column so that ties have a deterministic order.
      List<String> orderCandidates = new ArrayList<>(aggregations);
      orderCandidates.add("COUNT(*)");
      orderBy.add(pick(random, orderCandidates) + (random.nextBoolean() ? " ASC" : " DESC"));
      for (String key : keys) {
        orderBy.add(key + (random.nextBoolean() ? " ASC" : " DESC"));
      }
    }
    sql.append(" ORDER BY ").append(String.join(", ", orderBy)).append(" LIMIT ").append(pick(random, LIMITS));
    return new GeneratedQuery(sql.toString(), true);
  }

  private static String randomAggregation(Random random) {
    int choice = random.nextInt(100);
    if (choice < 8) {
      return pick(random, UNSUPPORTED_AGGREGATIONS);
    }
    if (choice < 30) {
      return "COUNT(*)";
    }
    if (choice < 50) {
      return "SUM(" + pick(random, INT_COLUMNS) + ")";
    }
    if (choice < 70) {
      return "AVG(" + pick(random, INT_COLUMNS) + ")";
    }
    return "DISTINCTCOUNT(" + pick(random, DICTIONARY_COLUMNS) + ")";
  }

  /// Returns `""` or a ` WHERE ...` clause of one or two predicates.
  private static String randomFilter(Random random) {
    int choice = random.nextInt(100);
    if (choice < 45) {
      return "";
    }
    if (choice < 80) {
      return " WHERE " + randomPredicate(random);
    }
    return " WHERE " + randomPredicate(random) + (random.nextBoolean() ? " AND " : " OR ") + randomPredicate(random);
  }

  private static String randomPredicate(Random random) {
    switch (random.nextInt(12)) {
      case 0:
        // Sorted column: range and equality, including bounds outside the values (all / none match).
        return pick(random, "sortedCol < ", "sortedCol >= ", "sortedCol = ") + pick(random, -5, 0, 3, 50, 97, 99, 200);
      case 1:
        int low = random.nextInt(110) - 5;
        return "sortedCol BETWEEN " + low + " AND " + (low + random.nextInt(60));
      case 2:
        // Inverted index.
        return pick(random, "invCol = ", "invCol <> ", "invCol < ") + random.nextInt(105);
      case 3:
        return "invCol" + pick(random, " IN (", " NOT IN (") + random.nextInt(100) + ", " + random.nextInt(100) + ", "
            + random.nextInt(100) + ")";
      case 4:
        return pick(random, "dStr = ", "dStr <> ", "dStr > ") + "'s" + random.nextInt(22) + "'";
      case 5:
        return "dStr IN ('s" + random.nextInt(20) + "', 's" + random.nextInt(20) + "')";
      case 6:
        // Dictionary scan on a metric with extreme values.
        return "metric " + pick(random, ">", "<=", "<>", ">=") + " " + pick(random, Integer.MIN_VALUE, -1001, -500,
            0, 500, 1000, Integer.MAX_VALUE - 100, Integer.MAX_VALUE);
      case 7:
        // Raw (no dictionary) scan; inclusive bounds, as the raw evaluator rejects empty ranges such as < MIN_VALUE.
        return "rawMetric " + pick(random, ">=", "<=") + " " + pick(random, Integer.MIN_VALUE, -1_000_000_000, 0,
            1_000_000_000, Integer.MAX_VALUE);
      case 8:
        return pick(random, "dLong >= ", "dLong < ") + pick(random, -5_000_000_000L, 0L, 5_000_000_010L);
      case 9:
        return pick(random, "dDouble < ", "dFloat >= ") + pick(random, -10, -1, 0, 1, 2.5, 100);
      case 10:
        return pick(random, "dBool = true", "dBool = false", "dConst = 'k'", "dConst <> 'k'", "dConst = 'z'");
      default:
        return random.nextBoolean() ? "dStrMed = 'm" + random.nextInt(400) + "'"
            : pick(random, "dIntMed < ", "dInt > ") + (random.nextInt(1600) - 30);
    }
  }

  /// Query shapes the GPU never runs; they must stay on the CPU with unchanged results.
  private static String unsupportedShape(Random random, String where) {
    int limit = pick(random, LIMITS);
    switch (random.nextInt(5)) {
      case 0:
        return "SELECT dStr, metric, rawMetric FROM testTable" + where + " ORDER BY dStr, metric, rawMetric LIMIT "
            + limit;
      case 1:
        return "SELECT DISTINCT dStr, dInt FROM testTable" + where + " ORDER BY dStr, dInt LIMIT " + limit;
      case 2:
        return "SELECT COUNT(*) FROM testTable" + where;
      case 3:
        return "SELECT dStr, COUNT(*) FILTER (WHERE metric > 0), SUM(metric) FROM testTable" + where
            + " GROUP BY dStr ORDER BY dStr LIMIT " + limit;
      default:
        return "SELECT dStr, COUNT(*) FROM testTable" + where + " GROUP BY dStr LIMIT 0";
    }
  }

  @SafeVarargs
  private static <T> T pick(Random random, T... values) {
    return values[random.nextInt(values.length)];
  }

  private static <T> T pick(Random random, List<T> values) {
    return values.get(random.nextInt(values.size()));
  }

  private static int pick(Random random, int[] values) {
    return values[random.nextInt(values.length)];
  }

  /// Server (over all segments) and broker; captures the GPU stats when the plan was routed to a GPU backend.
  private Execution execute(String query, PlanMaker planMaker) {
    PinotQuery pinotQuery = CalciteSqlParser.compileToPinotQuery(query);
    QueryContext queryContext = QueryContextConverterUtils.getQueryContext(pinotQuery);
    queryContext.setEndTimeMs(System.currentTimeMillis() + 60_000);
    List<SegmentContext> segmentContexts = new ArrayList<>();
    for (IndexSegment segment : _indexSegments) {
      segmentContexts.add(new SegmentContext(segment));
    }
    byte[] dataTable;
    GpuQueryStats stats = null;
    try (QueryThreadContext ignore = QueryThreadContext.openForSseTest()) {
      Plan plan = planMaker.makeInstancePlan(segmentContexts, queryContext, EXECUTOR_SERVICE);
      dataTable = plan.execute().toDataTable().toBytes();
      if (plan instanceof GpuInstancePlan) {
        stats = ((GpuInstancePlan) plan).getStats();
      }
    } catch (Exception e) {
      throw new RuntimeException("seed=" + _seed + " query failed on the server: " + query, e);
    }
    BrokerRequest brokerRequest = CalciteSqlCompiler.convertToBrokerRequest(pinotQuery);
    try {
      // Mutable: the broker drops data tables without rows.
      Map<ServerRoutingInstance, DataTable> dataTables = new HashMap<>();
      dataTables.put(new ServerRoutingInstance("localhost", 1234, TableType.OFFLINE),
          DataTableFactory.getDataTable(dataTable));
      return new Execution(reduceOnDataTable(brokerRequest, brokerRequest, dataTables), stats);
    } catch (Exception e) {
      throw new RuntimeException("seed=" + _seed + " query failed on the broker: " + query, e);
    }
  }

  /// Returns a description of the first difference, or `null` when the responses match.
  @Nullable
  private static String compare(BrokerResponseNative actual, BrokerResponseNative expected, boolean ordered) {
    if (!expected.getExceptions().isEmpty() || !actual.getExceptions().isEmpty()) {
      if (expected.getExceptions().size() != actual.getExceptions().size()) {
        return "exceptions differ: cpu=" + expected.getExceptions() + " gpu=" + actual.getExceptions();
      }
      return null;
    }
    if (!actual.getResultTable().getDataSchema().equals(expected.getResultTable().getDataSchema())) {
      return "schema differs: cpu=" + expected.getResultTable().getDataSchema() + " gpu="
          + actual.getResultTable().getDataSchema();
    }
    List<Object[]> actualRows = new ArrayList<>(actual.getResultTable().getRows());
    List<Object[]> expectedRows = new ArrayList<>(expected.getResultTable().getRows());
    if (!ordered) {
      Comparator<Object[]> byText = Comparator.comparing(Arrays::deepToString);
      actualRows.sort(byText);
      expectedRows.sort(byText);
    }
    int numRows = Math.min(actualRows.size(), expectedRows.size());
    for (int i = 0; i < numRows; i++) {
      if (!Arrays.deepEquals(actualRows.get(i), expectedRows.get(i))) {
        return "row " + i + " differs: cpu=" + Arrays.deepToString(expectedRows.get(i)) + " gpu="
            + Arrays.deepToString(actualRows.get(i));
      }
    }
    if (actualRows.size() != expectedRows.size()) {
      return "row count differs: cpu=" + expectedRows.size() + " gpu=" + actualRows.size();
    }
    String stats = compareStat("numDocsScanned", actual.getNumDocsScanned(), expected.getNumDocsScanned());
    stats = stats != null ? stats : compareStat("numEntriesScannedInFilter", actual.getNumEntriesScannedInFilter(),
        expected.getNumEntriesScannedInFilter());
    stats = stats != null ? stats : compareStat("numEntriesScannedPostFilter",
        actual.getNumEntriesScannedPostFilter(), expected.getNumEntriesScannedPostFilter());
    stats = stats != null ? stats : compareStat("numSegmentsMatched", actual.getNumSegmentsMatched(),
        expected.getNumSegmentsMatched());
    stats = stats != null ? stats : compareStat("numGroupsLimitReached", actual.isNumGroupsLimitReached(),
        expected.isNumGroupsLimitReached());
    stats = stats != null ? stats : compareStat("numGroupsWarningLimitReached",
        actual.isNumGroupsWarningLimitReached(), expected.isNumGroupsWarningLimitReached());
    return stats != null ? stats : compareStat("totalDocs", actual.getTotalDocs(), expected.getTotalDocs());
  }

  @Nullable
  private static String compareStat(String name, Object actual, Object expected) {
    return actual.equals(expected) ? null : name + " differs: cpu=" + expected + " gpu=" + actual;
  }

  @AfterClass
  public void tearDown() {
    for (GpuPlanMaker planMaker : List.of(_javaPlanMaker, _failingPlanMaker, _corruptPlanMaker)) {
      planMaker.close();
    }
    for (IndexSegment segment : _indexSegments) {
      segment.destroy();
    }
    FileUtils.deleteQuietly(INDEX_DIR);
  }
}

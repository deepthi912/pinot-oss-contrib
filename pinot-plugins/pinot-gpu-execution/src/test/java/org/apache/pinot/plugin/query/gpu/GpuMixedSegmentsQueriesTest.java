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
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.core.query.request.context.utils.QueryContextConverterUtils;
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
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;


/// One query over segments that take different paths: two run on the GPU, one runs the regular CPU group-by
/// (its group-by column has no dictionary). The combined result must match the CPU plan.
public class GpuMixedSegmentsQueriesTest extends BaseQueriesTest {
  private static final File INDEX_DIR = new File(FileUtils.getTempDirectory(), "GpuMixedSegmentsQueriesTest");
  private static final String TABLE = "testTable";
  private static final int NUM_ROWS = 10_007;
  private static final Schema SCHEMA = new Schema.SchemaBuilder().setSchemaName(TABLE)
      .addSingleValueDimension("dimStr", DataType.STRING)
      .addMetric("metric", DataType.INT)
      .build();

  private final JavaGpuBackendProvider _provider =
      new JavaGpuBackendProvider(GpuBackendType.CUDA, true, Behavior.CORRECT);
  private final InstancePlanMakerImplV2 _cpuPlanMaker = new InstancePlanMakerImplV2();
  private GpuPlanMaker _gpuPlanMaker;
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
    TableConfig dictionary = new TableConfigBuilder(TableType.OFFLINE).setTableName(TABLE).build();
    TableConfig raw = new TableConfigBuilder(TableType.OFFLINE).setTableName(TABLE)
        .setNoDictionaryColumns(List.of("dimStr")).build();
    _indexSegments = List.of(buildSegment("gpu0", 0, dictionary), buildSegment("gpu1", 1, dictionary),
        buildSegment("cpu0", 2, raw));
    _gpuPlanMaker = new GpuPlanMaker(List.of(_provider));
    _gpuPlanMaker.init(new PinotConfiguration(
        Map.of(GpuExecutionConfig.ENABLED, "true", GpuExecutionConfig.MIN_ROWS, "0")));
  }

  private static ImmutableSegment buildSegment(String name, long seed, TableConfig tableConfig)
      throws Exception {
    Random random = new Random(seed);
    List<GenericRow> rows = new ArrayList<>(NUM_ROWS);
    for (int i = 0; i < NUM_ROWS; i++) {
      GenericRow row = new GenericRow();
      row.putValue("dimStr", "s" + random.nextInt(60));
      row.putValue("metric", random.nextInt(1000));
      rows.add(row);
    }
    SegmentGeneratorConfig config = new SegmentGeneratorConfig(tableConfig, SCHEMA);
    config.setTableName(TABLE);
    config.setSegmentName(name);
    config.setOutDir(INDEX_DIR.getPath());
    SegmentIndexCreationDriverImpl driver = new SegmentIndexCreationDriverImpl();
    driver.init(config, new GenericRowRecordReader(rows));
    driver.build();
    return ImmutableSegmentLoader.load(new File(INDEX_DIR, name), ReadMode.mmap);
  }

  @DataProvider
  public Object[][] queries() {
    return new Object[][]{
        {"SELECT dimStr, COUNT(*) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 100"},
        {"SELECT dimStr, COUNT(*) FROM testTable GROUP BY dimStr ORDER BY dimStr LIMIT 7"},
        {"SELECT dimStr, COUNT(*) FROM testTable GROUP BY dimStr ORDER BY dimStr DESC LIMIT 7"},
        {"SELECT dimStr, COUNT(*) FROM testTable GROUP BY dimStr ORDER BY COUNT(*) DESC, dimStr LIMIT 5"},
        {"SELECT dimStr, COUNT(*) FROM testTable WHERE metric > 300 GROUP BY dimStr ORDER BY dimStr LIMIT 100"}
    };
  }

  @Test(dataProvider = "queries")
  public void testMixedSegmentsMatchCpu(String query)
      throws Exception {
    BrokerResponseNative expected = getBrokerResponse(query, _cpuPlanMaker);
    BrokerResponseNative actual = getBrokerResponse(query, _gpuPlanMaker);
    assertTrue(actual.getExceptions().isEmpty(), "GPU failed: " + actual.getExceptions());
    assertEquals(actual.getResultTable().getRows().size(), expected.getResultTable().getRows().size(), query);
    for (int i = 0; i < expected.getResultTable().getRows().size(); i++) {
      assertEquals(actual.getResultTable().getRows().get(i), expected.getResultTable().getRows().get(i), query);
    }
    assertEquals(actual.getNumDocsScanned(), expected.getNumDocsScanned(), query);

    GpuQueryStats stats = executeAndGetStats(query);
    assertEquals(stats.getSegmentsOnGpu(), 2, stats.toString());
    assertEquals(stats.getSegmentsIneligible(), 1, stats.toString());
  }

  private GpuQueryStats executeAndGetStats(String query)
      throws Exception {
    QueryContext queryContext = QueryContextConverterUtils.getQueryContext(query);
    queryContext.setEndTimeMs(System.currentTimeMillis() + 60_000);
    List<SegmentContext> segmentContexts = new ArrayList<>();
    for (IndexSegment segment : _indexSegments) {
      segmentContexts.add(new SegmentContext(segment));
    }
    try (QueryThreadContext ignore = QueryThreadContext.openForSseTest()) {
      Plan plan = _gpuPlanMaker.makeInstancePlan(segmentContexts, queryContext, EXECUTOR_SERVICE);
      plan.execute();
      return ((GpuInstancePlan) plan).getStats();
    }
  }

  @AfterClass
  public void tearDown() {
    _gpuPlanMaker.getColumnCache().close();
    _gpuPlanMaker.getRouter().close();
    for (IndexSegment segment : _indexSegments) {
      segment.destroy();
    }
    FileUtils.deleteQuietly(INDEX_DIR);
  }
}

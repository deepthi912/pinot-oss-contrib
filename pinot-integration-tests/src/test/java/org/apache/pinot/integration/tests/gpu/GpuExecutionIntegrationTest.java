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
package org.apache.pinot.integration.tests.gpu;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.File;
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.apache.pinot.integration.tests.BaseClusterIntegrationTest;
import org.apache.pinot.integration.tests.ClusterIntegrationTestUtils;
import org.apache.pinot.plugin.query.gpu.GpuExecutionConfig;
import org.apache.pinot.plugin.query.gpu.GpuPlanMaker;
import org.apache.pinot.spi.config.table.TableConfig;
import org.apache.pinot.spi.data.Schema;
import org.apache.pinot.spi.env.PinotConfiguration;
import org.apache.pinot.spi.utils.CommonConstants.Server;
import org.apache.pinot.util.TestUtils;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;


/// GPU execution through a real cluster whose server is configured with [GpuPlanMaker]: plugin loading from
/// `plan.maker.class`, ServiceLoader backend discovery, `gpu.*` server config, and query options via the broker.
public class GpuExecutionIntegrationTest extends BaseClusterIntegrationTest {

  @Override
  protected void overrideServerConf(PinotConfiguration serverConf) {
    String prefix = Server.QUERY_EXECUTOR_CONFIG_PREFIX + ".";
    serverConf.setProperty(prefix + Server.PLAN_MAKER_CLASS, GpuPlanMaker.class.getName());
    serverConf.setProperty(prefix + GpuExecutionConfig.ENABLED, "true");
    serverConf.setProperty(prefix + GpuExecutionConfig.MIN_ROWS, "0");
  }

  @BeforeClass
  public void setUp()
      throws Exception {
    System.setProperty(TestGpuBackendProvider.ENABLED_PROPERTY, "true");
    TestUtils.ensureDirectoriesExistAndEmpty(_tempDir, _segmentDir, _tarDir);
    startZk();
    startController();
    startBroker();
    startServer();

    Schema schema = createSchema();
    addSchema(schema);
    TableConfig tableConfig = createOfflineTableConfig();
    addTableConfig(tableConfig);
    List<File> avroFiles = unpackAvroData(_tempDir);
    ClusterIntegrationTestUtils.buildSegmentsFromAvro(avroFiles, tableConfig, schema, 0, _segmentDir, _tarDir);
    uploadSegments(getTableName(), _tarDir);
    waitForAllDocsLoaded(600_000L);
  }

  @DataProvider
  public Object[][] queries() {
    return new Object[][]{
        {"SELECT Carrier, COUNT(*) FROM mytable GROUP BY Carrier ORDER BY Carrier LIMIT 100"},
        {"SELECT DayOfWeek, COUNT(*) FROM mytable GROUP BY DayOfWeek ORDER BY COUNT(*) DESC, DayOfWeek LIMIT 10"},
        {"SELECT OriginState, COUNT(*) FROM mytable WHERE DayOfWeek > 2 GROUP BY OriginState "
            + "ORDER BY OriginState LIMIT 100"},
        {"SELECT Carrier, SUM(Distance), AVG(ActualElapsedTime), DISTINCTCOUNT(Origin) FROM mytable "
            + "GROUP BY Carrier ORDER BY Carrier LIMIT 100"},
        {"SELECT DayOfWeek, DISTINCTCOUNT(Dest) FROM mytable GROUP BY DayOfWeek "
            + "ORDER BY DISTINCTCOUNT(Dest) DESC, DayOfWeek LIMIT 3"},
        {"SELECT SUM(Distance), AVG(CRSDepTime), DISTINCTCOUNT(FlightNum) FROM mytable WHERE DayOfWeek > 2"}
    };
  }

  @Test(dataProvider = "queries")
  public void testGpuResultsMatchCpu(String query)
      throws Exception {
    JsonNode cpu = postQuery("SET useMultistageEngine = false; SET useGpu = false; " + query);
    int callsBefore = TestGpuBackendProvider.getNumGroupByCalls();
    JsonNode gpu = postQuery("SET useMultistageEngine = false; SET useGpu = true; " + query);
    assertTrue(gpu.get("exceptions").isEmpty(), gpu.toString());
    assertEquals(gpu.get("resultTable").get("rows"), cpu.get("resultTable").get("rows"), query);
    assertEquals(gpu.get("numDocsScanned"), cpu.get("numDocsScanned"), query);
    assertTrue(TestGpuBackendProvider.getNumGroupByCalls() > callsBefore, "GPU backend was not called: " + query);
  }

  @Test(dataProvider = "queries")
  public void testExplainShowsGpuOperator(String query)
      throws Exception {
    JsonNode plan = postQuery("SET useMultistageEngine = false; SET useGpu = true; EXPLAIN PLAN FOR " + query);
    String operator = query.contains("GROUP BY") ? "GPU_GROUP_BY" : "GPU_AGGREGATE";
    assertTrue(plan.toString().contains(operator), plan.toString());
  }

  @Test(dataProvider = "queries")
  public void testMultiStageEngineStaysOnCpu(String query)
      throws Exception {
    JsonNode cpu = postQuery("SET useMultistageEngine = true; SET useGpu = false; " + query);
    int callsBefore = TestGpuBackendProvider.getNumGroupByCalls();
    JsonNode mse = postQuery("SET useMultistageEngine = true; SET useGpu = true; " + query);
    assertTrue(mse.get("exceptions").isEmpty(), mse.toString());
    assertEquals(mse.get("resultTable").get("rows"), cpu.get("resultTable").get("rows"), query);
    assertEquals(TestGpuBackendProvider.getNumGroupByCalls(), callsBefore, "MSE reached the GPU: " + query);
  }

  @AfterClass
  public void tearDown()
      throws Exception {
    dropOfflineTable(getTableName());
    stopServer();
    stopBroker();
    stopController();
    stopZk();
    FileUtils.deleteDirectory(_tempDir);
    System.clearProperty(TestGpuBackendProvider.ENABLED_PROPERTY);
  }
}

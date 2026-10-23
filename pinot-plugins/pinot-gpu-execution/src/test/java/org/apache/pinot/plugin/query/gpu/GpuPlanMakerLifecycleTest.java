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

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.apache.pinot.plugin.query.gpu.JavaGpuBackendProvider.Behavior;
import org.apache.pinot.plugin.query.gpu.JavaGpuBackendProvider.JavaColumn;
import org.apache.pinot.segment.spi.IndexSegment;
import org.apache.pinot.spi.env.PinotConfiguration;
import org.testng.annotations.Test;

import static org.mockito.Mockito.mock;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotSame;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;


public class GpuPlanMakerLifecycleTest {
  private static final PinotConfiguration CONFIG =
      new PinotConfiguration(Map.of(GpuExecutionConfig.ENABLED, "true", GpuExecutionConfig.MIN_ROWS, "0"));

  @Test
  public void testReInitClosesThePreviousBackend() {
    JavaGpuBackendProvider provider = new JavaGpuBackendProvider(GpuBackendType.METAL, true, Behavior.CORRECT);
    GpuPlanMaker planMaker = new GpuPlanMaker(List.of(provider));
    planMaker.init(CONFIG);
    GpuBackendRouter first = planMaker.getRouter();
    assertTrue(first.decide(GpuQueryOptions.NONE, 1).runsOnGpu());
    planMaker.init(CONFIG);
    assertEquals(provider.getNumBackendCloses(), 1);
    assertNotSame(planMaker.getRouter(), first);
    assertTrue(planMaker.getRouter().decide(GpuQueryOptions.NONE, 1).runsOnGpu());
    planMaker.close();
    assertEquals(provider.getNumBackendCloses(), 2);
  }

  @Test
  public void testCloseWhileALeaseIsHeld() {
    JavaGpuBackendProvider provider = new JavaGpuBackendProvider(GpuBackendType.METAL, true, Behavior.CORRECT);
    GpuPlanMaker planMaker = new GpuPlanMaker(List.of(provider));
    planMaker.init(CONFIG);
    GpuBackendRouter router = planMaker.getRouter();
    GpuColumnCache cache = planMaker.getColumnCache();
    GpuBackend backend = router.decide(GpuQueryOptions.NONE, 1).getBackend();
    long instanceId = cache.getOrAssignInstanceId(mock(IndexSegment.class));
    GpuColumnCache.Lease lease = cache.acquire("k", instanceId, 40, TimeUnit.SECONDS.toNanos(10),
        () -> backend.uploadIntColumn(new int[10], 10));
    JavaColumn column = (JavaColumn) lease.getColumn();

    planMaker.close();
    planMaker.close();
    assertNull(planMaker.getRouter());
    assertNull(planMaker.getColumnCache());
    assertEquals(provider.getNumBackendCloses(), 1);
    assertFalse(router.decide(GpuQueryOptions.NONE, 1).runsOnGpu());
    assertFalse(column.isClosed(), "a leased column must outlive the close");
    lease.close();
    assertTrue(column.isClosed());
  }
}

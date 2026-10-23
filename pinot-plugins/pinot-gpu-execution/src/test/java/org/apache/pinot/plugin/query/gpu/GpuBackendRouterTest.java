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
import org.apache.pinot.plugin.query.gpu.GpuBackendRouter.Decision;
import org.apache.pinot.plugin.query.gpu.GpuBackendRouter.Reason;
import org.apache.pinot.plugin.query.gpu.JavaGpuBackendProvider.Behavior;
import org.apache.pinot.spi.env.PinotConfiguration;
import org.testng.annotations.Test;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;


public class GpuBackendRouterTest {
  private static final GpuQueryOptions FORCE = new GpuQueryOptions(true, null);

  private static GpuExecutionConfig enabled(long minRows) {
    return new GpuExecutionConfig(true, GpuExecutionConfig.DEFAULT_BACKENDS, minRows,
        GpuExecutionConfig.DEFAULT_CACHE_MAX_BYTES);
  }

  private static GpuBackendProvider provider(GpuBackendType type, boolean available) {
    return new JavaGpuBackendProvider(type, available, Behavior.CORRECT);
  }

  @Test
  public void testPrefersCudaThenMetalThenCpu() {
    try (GpuBackendRouter router = new GpuBackendRouter(enabled(0),
        List.of(provider(GpuBackendType.METAL, true), provider(GpuBackendType.CUDA, true)))) {
      assertEquals(router.decide(GpuQueryOptions.NONE, 1).getBackend().getType(), GpuBackendType.CUDA);
    }
    try (GpuBackendRouter router = new GpuBackendRouter(enabled(0),
        List.of(provider(GpuBackendType.CUDA, false), provider(GpuBackendType.METAL, true)))) {
      Decision decision = router.decide(GpuQueryOptions.NONE, 1);
      assertEquals(decision.getBackend().getType(), GpuBackendType.METAL);
      assertEquals(decision.getSkipped(), List.of("cuda: test device missing"));
    }
    try (GpuBackendRouter router = new GpuBackendRouter(enabled(0), List.of(provider(GpuBackendType.METAL, false)))) {
      Decision decision = router.decide(GpuQueryOptions.NONE, 1);
      assertNull(decision.getBackend());
      assertEquals(decision.getReason(), Reason.NO_GPU_AVAILABLE);
      assertEquals(decision.getSkipped(),
          List.of("cuda: no provider on the classpath", "metal: test device missing"));
    }
  }

  @Test
  public void testConfigIsTheMasterSwitch() {
    try (GpuBackendRouter router = new GpuBackendRouter(GpuExecutionConfig.disabled(),
        List.of(provider(GpuBackendType.CUDA, true)))) {
      Decision decision = router.decide(new GpuQueryOptions(true, GpuBackendType.CUDA), 1);
      assertFalse(decision.runsOnGpu());
      assertEquals(decision.getReason(), Reason.DISABLED_BY_CONFIG);
    }
  }

  @Test
  public void testQueryOptionsSteerWithinTheConfig() {
    try (GpuBackendRouter router = new GpuBackendRouter(enabled(1_000_000),
        List.of(provider(GpuBackendType.CUDA, true), provider(GpuBackendType.METAL, true)))) {
      assertEquals(router.decide(GpuQueryOptions.NONE, 10).getReason(), Reason.BELOW_MIN_ROWS);
      assertEquals(router.decide(GpuQueryOptions.NONE, 2_000_000).getReason(), Reason.GPU_SELECTED);
      assertEquals(router.decide(FORCE, 10).getReason(), Reason.GPU_FORCED);
      assertEquals(router.decide(new GpuQueryOptions(false, null), 2_000_000).getReason(), Reason.DISABLED_BY_QUERY);
      assertEquals(router.decide(new GpuQueryOptions(true, GpuBackendType.METAL), 10).getBackend().getType(),
          GpuBackendType.METAL);
      assertEquals(router.decide(new GpuQueryOptions(true, GpuBackendType.CPU), 10).getReason(),
          Reason.CPU_REQUESTED);
    }
  }

  @Test
  public void testUnavailablePreferredBackendFallsBackAlongTheOrder() {
    try (GpuBackendRouter router = new GpuBackendRouter(enabled(0), List.of(provider(GpuBackendType.METAL, true)))) {
      Decision decision = router.decide(new GpuQueryOptions(null, GpuBackendType.CUDA), 1);
      assertEquals(decision.getBackend().getType(), GpuBackendType.METAL);
      assertTrue(decision.getSkipped().get(0).startsWith("cuda:"));
    }
  }

  @Test
  public void testCreationFailureMarksBackendUnavailable() {
    GpuBackendProvider broken = new JavaGpuBackendProvider(GpuBackendType.CUDA, true, Behavior.CORRECT) {
      @Override
      public GpuBackend create() {
        throw new UnsatisfiedLinkError("libcudf.so: cannot open shared object file");
      }
    };
    try (GpuBackendRouter router = new GpuBackendRouter(enabled(0),
        List.of(broken, provider(GpuBackendType.METAL, true)))) {
      assertEquals(router.decide(GpuQueryOptions.NONE, 1).getBackend().getType(), GpuBackendType.METAL);
      assertFalse(router.getAvailability().get(GpuBackendType.CUDA).isAvailable());
      assertEquals(router.decide(GpuQueryOptions.NONE, 1).getBackend().getType(), GpuBackendType.METAL);
    }
  }

  @Test
  public void testQueryCannotPickBackendExcludedByServer() {
    GpuExecutionConfig metalOnly =
        new GpuExecutionConfig(true, List.of(GpuBackendType.METAL), 0, GpuExecutionConfig.DEFAULT_CACHE_MAX_BYTES);
    try (GpuBackendRouter router = new GpuBackendRouter(metalOnly,
        List.of(provider(GpuBackendType.CUDA, true), provider(GpuBackendType.METAL, true)))) {
      GpuBackendRouter.Decision decision =
          router.decide(GpuQueryOptions.fromQueryOptions(Map.of("gpuBackend", "cuda")), 1);
      assertEquals(decision.getBackend().getType(), GpuBackendType.METAL);
      assertTrue(decision.getSkipped().get(0).startsWith("cuda: not in"), decision.getSkipped().toString());
    }
  }

  @Test
  public void testParsesConfigAndQueryOptions() {
    GpuExecutionConfig config = GpuExecutionConfig.fromQueryExecutorConfig(new PinotConfiguration(Map.of(
        GpuExecutionConfig.ENABLED, "true", GpuExecutionConfig.BACKENDS, "metal, cpu, cuda",
        GpuExecutionConfig.MIN_ROWS, "5000")));
    assertTrue(config.isEnabled());
    assertEquals(config.getBackends(), List.of(GpuBackendType.METAL, GpuBackendType.CUDA, GpuBackendType.CPU));
    assertEquals(config.getMinRows(), 5000);
    assertEquals(GpuExecutionConfig.fromQueryExecutorConfig(new PinotConfiguration(
        Map.of(GpuExecutionConfig.CACHE_MAX_BYTES, "4096"))).getCacheMaxBytes(), 4096);
    assertThrows(IllegalArgumentException.class, () -> GpuExecutionConfig.fromQueryExecutorConfig(
        new PinotConfiguration(Map.of(GpuExecutionConfig.MIN_ROWS, "-1"))));
    assertThrows(IllegalArgumentException.class, () -> GpuExecutionConfig.fromQueryExecutorConfig(
        new PinotConfiguration(Map.of(GpuExecutionConfig.CACHE_MAX_BYTES, "-1"))));
    assertThrows(IllegalArgumentException.class, () -> GpuExecutionConfig.fromQueryExecutorConfig(
        new PinotConfiguration(Map.of(GpuExecutionConfig.BACKENDS, "rocm"))));
    GpuExecutionConfig defaults = GpuExecutionConfig.fromQueryExecutorConfig(new PinotConfiguration(Map.of()));
    assertFalse(defaults.isEnabled());
    assertEquals(defaults.getBackends(), GpuExecutionConfig.DEFAULT_BACKENDS);
    assertEquals(defaults.getMinRows(), GpuExecutionConfig.DEFAULT_MIN_ROWS);
    assertEquals(defaults.getCacheMaxBytes(), GpuExecutionConfig.DEFAULT_CACHE_MAX_BYTES);

    GpuQueryOptions options = GpuQueryOptions.fromQueryOptions(Map.of("USEGPU", "true", "gpuBackend", "Metal"));
    assertEquals(options.getUseGpu(), Boolean.TRUE);
    assertEquals(options.getPreferredBackend(), GpuBackendType.METAL);
    assertThrows(IllegalArgumentException.class, () -> GpuQueryOptions.fromQueryOptions(Map.of("useGpu", "maybe")));
    assertThrows(IllegalArgumentException.class,
        () -> GpuQueryOptions.fromQueryOptions(Map.of("gpuBackend", "rocm")));
  }
}

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

import java.util.Map;
import org.apache.pinot.plugin.query.gpu.metal.MetalGpuBackendProvider;
import org.apache.pinot.spi.env.PinotConfiguration;
import org.testng.SkipException;

import static org.testng.Assert.fail;


/// Metal tests skip where Metal is unavailable, and fail there instead with `-Dpinot.gpu.metal=true`.
final class MetalTestSupport {
  static final String METAL_PROPERTY = "pinot.gpu.metal";

  private MetalTestSupport() {
  }

  static void requireMetal() {
    GpuBackendProvider.Availability availability = new MetalGpuBackendProvider().probe();
    if (!availability.isAvailable()) {
      if (Boolean.getBoolean(METAL_PROPERTY)) {
        fail("-D" + METAL_PROPERTY + "=true but the Metal backend is unavailable: " + availability);
      }
      throw new SkipException("Metal backend not available: " + availability);
    }
  }

  /// A [GpuPlanMaker] with discovered backends (Metal here) and no row minimum; the caller closes it.
  static GpuPlanMaker metalPlanMaker() {
    requireMetal();
    GpuPlanMaker planMaker = new GpuPlanMaker();
    planMaker.init(new PinotConfiguration(
        Map.of(GpuExecutionConfig.ENABLED, "true", GpuExecutionConfig.MIN_ROWS, "0")));
    return planMaker;
  }
}

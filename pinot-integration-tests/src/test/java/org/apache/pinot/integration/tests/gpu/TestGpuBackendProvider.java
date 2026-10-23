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

import org.apache.pinot.plugin.query.gpu.GpuBackend;
import org.apache.pinot.plugin.query.gpu.GpuBackendProvider;
import org.apache.pinot.plugin.query.gpu.GpuBackendType;
import org.apache.pinot.plugin.query.gpu.JavaGpuBackendProvider;


/// ServiceLoader entry for the plugin's plain-Java test backend; available only while system property
/// [#ENABLED_PROPERTY] is true, so other tests are unaffected. Thread-safe.
public class TestGpuBackendProvider implements GpuBackendProvider {
  public static final String ENABLED_PROPERTY = "pinot.gpu.test.backend.enabled";
  private static final JavaGpuBackendProvider DELEGATE =
      new JavaGpuBackendProvider(GpuBackendType.CUDA, true, JavaGpuBackendProvider.Behavior.CORRECT);

  public static int getNumBackendCalls() {
    return DELEGATE.getNumBackendCalls();
  }

  @Override
  public GpuBackendType getType() {
    return GpuBackendType.CUDA;
  }

  @Override
  public Availability probe() {
    return Boolean.getBoolean(ENABLED_PROPERTY) ? DELEGATE.probe()
        : Availability.unavailable("integration test backend disabled");
  }

  @Override
  public GpuBackend create() {
    return DELEGATE.create();
  }
}

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

import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/// Per-query options: `useGpu` (true skips the `gpu.min.rows` gate, false forces the CPU) and `gpuBackend` (backend to
/// try first). Both only apply when the server enables GPU execution. An invalid value is ignored, with a warning
/// logged once, so it never fails a query. Immutable.
public final class GpuQueryOptions {
  private static final Logger LOGGER = LoggerFactory.getLogger(GpuQueryOptions.class);
  private static final AtomicBoolean INVALID_VALUE_LOGGED = new AtomicBoolean();
  public static final String USE_GPU = "useGpu";
  public static final String GPU_BACKEND = "gpuBackend";

  public static final GpuQueryOptions NONE = new GpuQueryOptions(null, null);

  @Nullable
  private final Boolean _useGpu;
  @Nullable
  private final GpuBackendType _preferredBackend;

  public GpuQueryOptions(@Nullable Boolean useGpu, @Nullable GpuBackendType preferredBackend) {
    _useGpu = useGpu;
    _preferredBackend = preferredBackend;
  }

  public static GpuQueryOptions fromQueryOptions(@Nullable Map<String, String> queryOptions) {
    if (queryOptions == null || queryOptions.isEmpty()) {
      return NONE;
    }
    String useGpu = null;
    String backend = null;
    for (Map.Entry<String, String> entry : queryOptions.entrySet()) {
      if (USE_GPU.equalsIgnoreCase(entry.getKey())) {
        useGpu = entry.getValue();
      } else if (GPU_BACKEND.equalsIgnoreCase(entry.getKey())) {
        backend = entry.getValue();
      }
    }
    if (useGpu == null && backend == null) {
      return NONE;
    }
    return new GpuQueryOptions(parseUseGpu(useGpu), parseBackend(backend));
  }

  @Nullable
  private static Boolean parseUseGpu(@Nullable String value) {
    if (value == null) {
      return null;
    }
    switch (value.trim().toLowerCase(Locale.ROOT)) {
      case "true":
        return Boolean.TRUE;
      case "false":
        return Boolean.FALSE;
      default:
        logInvalid(USE_GPU, value, "expected true or false");
        return null;
    }
  }

  @Nullable
  private static GpuBackendType parseBackend(@Nullable String value) {
    if (value == null) {
      return null;
    }
    try {
      return GpuBackendType.fromId(value);
    } catch (IllegalArgumentException e) {
      logInvalid(GPU_BACKEND, value, e.getMessage());
      return null;
    }
  }

  private static void logInvalid(String option, String value, String detail) {
    if (INVALID_VALUE_LOGGED.compareAndSet(false, true)) {
      LOGGER.warn("Ignoring invalid value for query option {}: '{}' ({}); further invalid values are logged at DEBUG",
          option, value, detail);
    } else {
      LOGGER.debug("Ignoring invalid value for query option {}: '{}' ({})", option, value, detail);
    }
  }

  @Nullable
  public Boolean getUseGpu() {
    return _useGpu;
  }

  @Nullable
  public GpuBackendType getPreferredBackend() {
    return _preferredBackend;
  }
}

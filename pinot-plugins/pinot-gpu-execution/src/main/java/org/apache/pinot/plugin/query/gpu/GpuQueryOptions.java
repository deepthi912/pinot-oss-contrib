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
import javax.annotation.Nullable;


/// Per-query GPU options, set like any other query option, e.g. `SET useGpu = true; SELECT ...`.
///
/// - `useGpu`: `true` uses the GPU even below the configured minimum row count, `false` keeps the query on the
///   CPU, absent lets the server config decide.
/// - `gpuBackend`: `cuda`, `metal` or `cpu`, tried first; an unavailable backend still falls back along the
///   configured order.
///
/// Neither option can enable GPU execution when [GpuExecutionConfig#isEnabled()] is `false`. Immutable.
public final class GpuQueryOptions {
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

  /// Option keys are matched case-insensitively.
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
    return new GpuQueryOptions(parseUseGpu(useGpu), backend != null ? GpuBackendType.fromId(backend) : null);
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
        throw new IllegalArgumentException("Invalid value for query option " + USE_GPU + ": '" + value
            + "', expected true or false");
    }
  }

  /// `null` when the option is absent.
  @Nullable
  public Boolean getUseGpu() {
    return _useGpu;
  }

  /// `null` when the option is absent.
  @Nullable
  public GpuBackendType getPreferredBackend() {
    return _preferredBackend;
  }
}

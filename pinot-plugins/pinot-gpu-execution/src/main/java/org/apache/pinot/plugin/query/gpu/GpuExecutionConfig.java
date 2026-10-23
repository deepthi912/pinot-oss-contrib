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

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import javax.annotation.Nullable;
import org.apache.pinot.spi.env.PinotConfiguration;


/// Server-level GPU execution settings under `pinot.server.query.executor.gpu.*`, read once when the plan maker is
/// initialized (changes need a server restart). `gpu.enabled` is the master switch: when false, no query option
/// enables the GPU. Immutable.
public final class GpuExecutionConfig {
  public static final String ENABLED = "gpu.enabled";
  public static final String BACKENDS = "gpu.backends";
  public static final String MIN_ROWS = "gpu.min.rows";
  public static final String CACHE_MAX_BYTES = "gpu.cache.max.bytes";

  public static final List<GpuBackendType> DEFAULT_BACKENDS =
      List.of(GpuBackendType.CUDA, GpuBackendType.METAL, GpuBackendType.CPU);
  public static final long DEFAULT_MIN_ROWS = 10_000_000L;
  public static final long DEFAULT_CACHE_MAX_BYTES = 2L * 1024 * 1024 * 1024;

  private final boolean _enabled;
  private final List<GpuBackendType> _backends;
  private final long _minRows;
  private final long _cacheMaxBytes;

  public GpuExecutionConfig(boolean enabled, List<GpuBackendType> backends, long minRows, long cacheMaxBytes) {
    if (minRows < 0) {
      throw new IllegalArgumentException(MIN_ROWS + " must be non-negative, got: " + minRows);
    }
    if (cacheMaxBytes < 0) {
      throw new IllegalArgumentException(CACHE_MAX_BYTES + " must be non-negative, got: " + cacheMaxBytes);
    }
    _enabled = enabled;
    _backends = withCpuLast(backends);
    _minRows = minRows;
    _cacheMaxBytes = cacheMaxBytes;
  }

  public static GpuExecutionConfig disabled() {
    return new GpuExecutionConfig(false, DEFAULT_BACKENDS, DEFAULT_MIN_ROWS, DEFAULT_CACHE_MAX_BYTES);
  }

  public static GpuExecutionConfig fromQueryExecutorConfig(PinotConfiguration queryExecutorConfig) {
    return fromValues(queryExecutorConfig.getProperty(ENABLED), queryExecutorConfig.getProperty(BACKENDS),
        queryExecutorConfig.getProperty(MIN_ROWS), queryExecutorConfig.getProperty(CACHE_MAX_BYTES));
  }

  private static GpuExecutionConfig fromValues(String enabled, String backends, String minRows,
      String cacheMaxBytes) {
    List<GpuBackendType> order = DEFAULT_BACKENDS;
    if (backends != null && !backends.isBlank()) {
      order = new ArrayList<>();
      for (String id : backends.split(",")) {
        if (!id.isBlank()) {
          order.add(GpuBackendType.fromId(id));
        }
      }
    }
    return new GpuExecutionConfig(parseEnabled(enabled), order,
        minRows != null ? Long.parseLong(minRows.trim()) : DEFAULT_MIN_ROWS,
        cacheMaxBytes != null ? Long.parseLong(cacheMaxBytes.trim()) : DEFAULT_CACHE_MAX_BYTES);
  }

  private static boolean parseEnabled(@Nullable String enabled) {
    if (enabled == null) {
      return false;
    }
    switch (enabled.trim().toLowerCase(Locale.ROOT)) {
      case "true":
        return true;
      case "false":
        return false;
      default:
        throw new IllegalArgumentException(ENABLED + " must be true or false, got: " + enabled);
    }
  }

  private static List<GpuBackendType> withCpuLast(List<GpuBackendType> backends) {
    List<GpuBackendType> order = new ArrayList<>();
    for (GpuBackendType type : backends) {
      if (type != GpuBackendType.CPU && !order.contains(type)) {
        order.add(type);
      }
    }
    order.add(GpuBackendType.CPU);
    return List.copyOf(order);
  }

  public boolean isEnabled() {
    return _enabled;
  }

  public List<GpuBackendType> getBackends() {
    return _backends;
  }

  public long getMinRows() {
    return _minRows;
  }

  public long getCacheMaxBytes() {
    return _cacheMaxBytes;
  }

  @Override
  public String toString() {
    return "GpuExecutionConfig{enabled=" + _enabled + ", backends=" + _backends + ", minRows=" + _minRows
        + ", cacheMaxBytes=" + _cacheMaxBytes + '}';
  }
}

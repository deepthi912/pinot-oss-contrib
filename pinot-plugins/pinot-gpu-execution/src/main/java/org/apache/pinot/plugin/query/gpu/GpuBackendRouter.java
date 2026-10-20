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
import java.util.EnumMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/// Chooses the backend for each query, in the configured order (CUDA, then Metal, then CPU by default), within what
/// the server config allows and as steered by query options.
///
/// Every backend is probed once at construction; unavailable ones are skipped with the reason recorded. A backend
/// that fails to be created is marked unavailable so later queries skip it cheaply. Each [Decision] carries which
/// backend was chosen, why, and every backend skipped on the way.
///
/// Thread-safe: decisions read a volatile config snapshot, and backends are created at most once. [#close()] must run
/// after every [GpuColumnCache] holding columns of these backends has been closed.
public class GpuBackendRouter implements AutoCloseable {
  private static final Logger LOGGER = LoggerFactory.getLogger(GpuBackendRouter.class);

  /// Why a query got the backend it got.
  public enum Reason {
    /// `gpu.enabled` is false: the master switch.
    DISABLED_BY_CONFIG,
    /// Query option `useGpu=false`.
    DISABLED_BY_QUERY,
    /// Fewer rows than `gpu.min.rows`, and the query did not set `useGpu=true`.
    BELOW_MIN_ROWS,
    /// A GPU backend was chosen automatically.
    GPU_SELECTED,
    /// A GPU backend was chosen because the query set `useGpu=true`.
    GPU_FORCED,
    /// The query asked for `gpuBackend=cpu`.
    CPU_REQUESTED,
    /// GPU execution was wanted but no GPU backend is available on this server.
    NO_GPU_AVAILABLE
  }

  /// The backend chosen for one query. [#getBackend()] is `null` when the query runs on the regular CPU plan.
  public static final class Decision {
    @Nullable
    private final GpuBackend _backend;
    private final Reason _reason;
    private final List<String> _skipped;

    Decision(@Nullable GpuBackend backend, Reason reason, List<String> skipped) {
      _backend = backend;
      _reason = reason;
      _skipped = List.copyOf(skipped);
    }

    @Nullable
    public GpuBackend getBackend() {
      return _backend;
    }

    public boolean runsOnGpu() {
      return _backend != null;
    }

    public Reason getReason() {
      return _reason;
    }

    /// Backends tried before the chosen one, each with the reason it was skipped.
    public List<String> getSkipped() {
      return _skipped;
    }

    @Override
    public String toString() {
      return "backend=" + (_backend != null ? _backend.getName() : GpuBackendType.CPU.getId()) + " reason=" + _reason
          + (_skipped.isEmpty() ? "" : " skipped=" + _skipped);
    }
  }

  private final Map<GpuBackendType, GpuBackendProvider> _providers = new EnumMap<>(GpuBackendType.class);
  private final Map<GpuBackendType, GpuBackendProvider.Availability> _availability = new ConcurrentHashMap<>();
  private final Map<GpuBackendType, GpuBackend> _backends = new ConcurrentHashMap<>();
  private volatile GpuExecutionConfig _config;

  public GpuBackendRouter(GpuExecutionConfig config, List<GpuBackendProvider> providers) {
    _config = config;
    for (GpuBackendProvider provider : providers) {
      if (provider.getType() != GpuBackendType.CPU) {
        _providers.putIfAbsent(provider.getType(), provider);
      }
    }
    for (GpuBackendType type : GpuBackendType.values()) {
      if (type == GpuBackendType.CPU) {
        continue;
      }
      GpuBackendProvider provider = _providers.get(type);
      GpuBackendProvider.Availability availability;
      if (provider == null) {
        availability = GpuBackendProvider.Availability.unavailable("no provider on the classpath");
      } else {
        try {
          availability = provider.probe();
        } catch (RuntimeException | LinkageError e) {
          LOGGER.warn("Probing GPU backend {} failed", type.getId(), e);
          availability = GpuBackendProvider.Availability.unavailable("probe failed: " + e);
        }
      }
      _availability.put(type, availability);
      LOGGER.info("GPU backend {}: {}", type.getId(), availability);
    }
    LOGGER.info("GPU execution config: {}", config);
  }

  public GpuExecutionConfig getConfig() {
    return _config;
  }

  public void setConfig(GpuExecutionConfig config) {
    LOGGER.info("Updating GPU execution config to: {}", config);
    _config = config;
  }

  /// Whether each GPU backend can be used on this server, and why.
  public Map<GpuBackendType, GpuBackendProvider.Availability> getAvailability() {
    return Map.copyOf(_availability);
  }

  /// Chooses the backend for a query over segments holding `numRows` documents in total.
  public Decision decide(GpuQueryOptions options, long numRows) {
    GpuExecutionConfig config = _config;
    if (!config.isEnabled()) {
      return new Decision(null, Reason.DISABLED_BY_CONFIG, List.of());
    }
    if (Boolean.FALSE.equals(options.getUseGpu())) {
      return new Decision(null, Reason.DISABLED_BY_QUERY, List.of());
    }
    boolean forced = Boolean.TRUE.equals(options.getUseGpu());
    if (!forced && numRows < config.getMinRows()) {
      return new Decision(null, Reason.BELOW_MIN_ROWS, List.of());
    }
    Set<GpuBackendType> order = new LinkedHashSet<>();
    if (options.getPreferredBackend() != null) {
      order.add(options.getPreferredBackend());
    }
    order.addAll(config.getBackends());

    List<String> skipped = new ArrayList<>();
    for (GpuBackendType type : order) {
      if (type == GpuBackendType.CPU) {
        Reason reason = options.getPreferredBackend() == GpuBackendType.CPU ? Reason.CPU_REQUESTED
            : Reason.NO_GPU_AVAILABLE;
        return new Decision(null, reason, skipped);
      }
      GpuBackendProvider.Availability availability = _availability.get(type);
      if (!availability.isAvailable()) {
        skipped.add(type.getId() + ": " + availability.getDetail());
        continue;
      }
      GpuBackend backend = getOrCreateBackend(type, skipped);
      if (backend != null) {
        return new Decision(backend, forced ? Reason.GPU_FORCED : Reason.GPU_SELECTED, skipped);
      }
    }
    // Unreachable: CPU is always last in the order.
    return new Decision(null, Reason.NO_GPU_AVAILABLE, skipped);
  }

  @Nullable
  private GpuBackend getOrCreateBackend(GpuBackendType type, List<String> skipped) {
    GpuBackend backend = _backends.get(type);
    if (backend != null) {
      return backend;
    }
    synchronized (_backends) {
      backend = _backends.get(type);
      if (backend != null) {
        return backend;
      }
      try {
        backend = _providers.get(type).create();
        _backends.put(type, backend);
        LOGGER.info("Created GPU backend: {}", backend.getName());
        return backend;
      } catch (RuntimeException | LinkageError e) {
        String detail = "creation failed: " + e;
        _availability.put(type, GpuBackendProvider.Availability.unavailable(detail));
        skipped.add(type.getId() + ": " + detail);
        LOGGER.warn("GPU backend {} is unavailable", type.getId(), e);
        return null;
      }
    }
  }

  @Override
  public void close() {
    for (GpuBackend backend : _backends.values()) {
      try {
        backend.close();
      } catch (RuntimeException e) {
        LOGGER.warn("Failed to close GPU backend: {}", backend.getName(), e);
      }
    }
    _backends.clear();
  }
}

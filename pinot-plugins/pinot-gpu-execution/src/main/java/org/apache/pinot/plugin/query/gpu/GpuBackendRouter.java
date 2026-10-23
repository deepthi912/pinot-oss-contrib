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


/// Chooses the backend for each query, in the configured order (CUDA, then Metal, then CPU by default), within what the
/// server config allows and as steered by query options. Thread-safe: decisions read a volatile config snapshot, and
/// backends are created at most once.
public class GpuBackendRouter implements AutoCloseable {
  private static final Logger LOGGER = LoggerFactory.getLogger(GpuBackendRouter.class);

  public enum Reason {
    DISABLED_BY_CONFIG,
    DISABLED_BY_QUERY,
    BELOW_MIN_ROWS,
    GPU_SELECTED,
    GPU_FORCED,
    CPU_REQUESTED,
    NO_GPU_AVAILABLE
  }

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

  public Map<GpuBackendType, GpuBackendProvider.Availability> getAvailability() {
    return Map.copyOf(_availability);
  }

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
    List<String> skipped = new ArrayList<>();
    Set<GpuBackendType> order = new LinkedHashSet<>();
    GpuBackendType preferred = options.getPreferredBackend();
    if (preferred != null) {
      // A query may only pick among the backends the server allows.
      if (config.getBackends().contains(preferred)) {
        order.add(preferred);
      } else {
        skipped.add(preferred.getId() + ": not in " + GpuExecutionConfig.BACKENDS);
      }
    }
    order.addAll(config.getBackends());

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

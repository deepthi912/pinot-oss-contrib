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

import com.google.common.annotations.VisibleForTesting;
import java.util.ArrayList;
import java.util.List;
import java.util.ServiceLoader;
import java.util.concurrent.ExecutorService;
import javax.annotation.Nullable;
import org.apache.pinot.core.plan.Plan;
import org.apache.pinot.core.plan.PlanNode;
import org.apache.pinot.core.plan.maker.InstancePlanMakerImplV2;
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.segment.spi.SegmentContext;
import org.apache.pinot.spi.env.PinotConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/// Server plan maker that runs supported single-stage aggregation queries on a GPU backend (CUDA, then Metal) and
/// everything else exactly as [InstancePlanMakerImplV2] does. Thread-safe: the per-query decision travels to
/// [#makeSegmentPlanNode] through a thread-local that is set and cleared around the synchronous planning in
/// [#makeInstancePlan].
public class GpuPlanMaker extends InstancePlanMakerImplV2 {
  private static final Logger LOGGER = LoggerFactory.getLogger(GpuPlanMaker.class);

  @Nullable
  private final List<GpuBackendProvider> _providers;
  private final ThreadLocal<ActiveQuery> _activeQuery = new ThreadLocal<>();
  private volatile GpuBackendRouter _router;
  private volatile GpuColumnCache _columnCache;

  public GpuPlanMaker() {
    this(null);
  }

  @VisibleForTesting
  GpuPlanMaker(@Nullable List<GpuBackendProvider> providers) {
    _providers = providers;
  }

  @Override
  public void init(PinotConfiguration queryExecutorConfig) {
    super.init(queryExecutorConfig);
    GpuExecutionConfig config = GpuExecutionConfig.fromQueryExecutorConfig(queryExecutorConfig);
    _router = new GpuBackendRouter(config, _providers != null ? _providers : discoverProviders());
    _columnCache = new GpuColumnCache(config.getCacheMaxBytes());
  }

  private static List<GpuBackendProvider> discoverProviders() {
    List<GpuBackendProvider> providers = new ArrayList<>();
    for (GpuBackendProvider provider : ServiceLoader.load(GpuBackendProvider.class,
        GpuPlanMaker.class.getClassLoader())) {
      providers.add(provider);
    }
    return providers;
  }

  @Nullable
  public GpuBackendRouter getRouter() {
    return _router;
  }

  @Nullable
  public GpuColumnCache getColumnCache() {
    return _columnCache;
  }

  // MSE leaf and streaming queries use makeStreamingInstancePlan (inherited), so they never reach the GPU.
  @Override
  public Plan makeInstancePlan(List<SegmentContext> segmentContexts, QueryContext queryContext,
      ExecutorService executorService) {
    GpuBackendRouter router = _router;
    if (router == null || !router.getConfig().isEnabled()) {
      return super.makeInstancePlan(segmentContexts, queryContext, executorService);
    }
    long numRows = 0;
    for (SegmentContext segmentContext : segmentContexts) {
      numRows += segmentContext.getIndexSegment().getSegmentMetadata().getTotalDocs();
    }
    GpuBackendRouter.Decision decision =
        router.decide(GpuQueryOptions.fromQueryOptions(queryContext.getQueryOptions()), numRows);
    LOGGER.debug("GPU decision for table: {}, segments: {}, rows: {}: {}", queryContext.getTableName(),
        segmentContexts.size(), numRows, decision);
    if (!decision.runsOnGpu()) {
      return super.makeInstancePlan(segmentContexts, queryContext, executorService);
    }
    GpuQueryStats stats = new GpuQueryStats(decision.getBackend().getName(), segmentContexts.size());
    _activeQuery.set(new ActiveQuery(decision.getBackend(), stats));
    try {
      return new GpuInstancePlan(super.makeInstancePlan(segmentContexts, queryContext, executorService), stats);
    } finally {
      _activeQuery.remove();
    }
  }

  @Override
  public PlanNode makeSegmentPlanNode(SegmentContext segmentContext, QueryContext queryContext) {
    ActiveQuery activeQuery = _activeQuery.get();
    if (activeQuery != null) {
      String reason = GpuQueryEligibility.checkQuery(queryContext);
      if (reason == null) {
        reason = GpuQueryEligibility.checkSegment(segmentContext.getIndexSegment(), queryContext);
      }
      if (reason == null) {
        return new GpuGroupByPlanNode(segmentContext, queryContext, activeQuery._backend, _columnCache,
            activeQuery._stats);
      }
      activeQuery._stats.recordIneligible();
      LOGGER.debug("Segment: {} stays on CPU: {}", segmentContext.getIndexSegment().getSegmentName(), reason);
    }
    return super.makeSegmentPlanNode(segmentContext, queryContext);
  }

  private static final class ActiveQuery {
    final GpuBackend _backend;
    final GpuQueryStats _stats;

    ActiveQuery(GpuBackend backend, GpuQueryStats stats) {
      _backend = backend;
      _stats = stats;
    }
  }
}

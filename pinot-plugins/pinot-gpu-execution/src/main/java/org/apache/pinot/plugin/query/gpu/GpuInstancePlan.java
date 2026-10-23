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

import java.util.concurrent.TimeoutException;
import org.apache.pinot.core.operator.blocks.InstanceResponseBlock;
import org.apache.pinot.core.plan.Plan;
import org.apache.pinot.core.plan.PlanNode;
import org.apache.pinot.core.util.trace.TraceContext;
import org.apache.pinot.spi.query.QueryThreadContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/// Instance plan of a query routed to a GPU backend. Executes the regular plan, then reports where this server's
/// segments actually ran ([GpuQueryStats]): as a trace entry and an INFO log line for traced queries, at DEBUG
/// otherwise. Not thread-safe.
public class GpuInstancePlan implements Plan {
  private static final Logger LOGGER = LoggerFactory.getLogger(GpuInstancePlan.class);
  public static final String TRACE_KEY = "GPU_SUMMARY";

  private final Plan _delegate;
  private final GpuQueryStats _stats;

  public GpuInstancePlan(Plan delegate, GpuQueryStats stats) {
    _delegate = delegate;
    _stats = stats;
  }

  public GpuQueryStats getStats() {
    return _stats;
  }

  private static String requestId() {
    try {
      return Long.toString(QueryThreadContext.get().getExecutionContext().getRequestId());
    } catch (RuntimeException e) {
      return "unknown";
    }
  }

  @Override
  public PlanNode getPlanNode() {
    return _delegate.getPlanNode();
  }

  @Override
  public InstanceResponseBlock execute()
      throws TimeoutException {
    InstanceResponseBlock response = _delegate.execute();
    if (TraceContext.traceEnabled()) {
      TraceContext.logInfo(TRACE_KEY, _stats.toString());
      LOGGER.info("{} requestId={} {}", TRACE_KEY, requestId(), _stats);
    } else if (LOGGER.isDebugEnabled()) {
      LOGGER.debug("{} requestId={} {}", TRACE_KEY, requestId(), _stats);
    }
    return response;
  }
}

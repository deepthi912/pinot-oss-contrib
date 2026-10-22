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

import javax.annotation.Nullable;


/// Per-group row counts and exact 64-bit sums from [GpuBackend#aggregate], plus the device-measured execution time
/// (`-1` when not measured). Immutable.
public final class GpuAggregationResult {
  private final long[] _counts;
  @Nullable
  private final long[] _sums;
  private final long _deviceTimeNs;

  public GpuAggregationResult(long[] counts, @Nullable long[] sums, long deviceTimeNs) {
    _counts = counts;
    _sums = sums;
    _deviceTimeNs = deviceTimeNs;
  }

  public long[] getCounts() {
    return _counts;
  }

  /// `null` when no value column was aggregated.
  @Nullable
  public long[] getSums() {
    return _sums;
  }

  public long getDeviceTimeNs() {
    return _deviceTimeNs;
  }
}

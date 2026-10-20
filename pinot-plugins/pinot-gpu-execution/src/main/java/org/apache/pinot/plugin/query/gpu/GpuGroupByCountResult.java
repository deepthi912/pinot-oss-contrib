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

/// Result of [GpuBackend#groupByCount]: the per-group counts plus the device's own execution time.
///
/// The device time comes from the GPU (e.g. Metal command buffer start/end timestamps), not from a host clock, so a
/// positive value proves the work executed on the device. Backends that cannot measure it report `-1`. Immutable.
public final class GpuGroupByCountResult {
  private final long[] _counts;
  private final long _deviceTimeNs;

  public GpuGroupByCountResult(long[] counts, long deviceTimeNs) {
    _counts = counts;
    _deviceTimeNs = deviceTimeNs;
  }

  /// Number of matching documents per dictionary id.
  public long[] getCounts() {
    return _counts;
  }

  /// Time the device spent executing the work, in nanoseconds, or `-1` when not measured.
  public long getDeviceTimeNs() {
    return _deviceTimeNs;
  }
}

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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;


/// Where one query's segments actually executed, accumulated across the combine's worker threads.
///
/// Reported once per query (trace entry `GPU_SUMMARY` and a debug log line) so a client can verify that every
/// segment ran on the GPU, with device-measured kernel time, rather than only that the GPU path was planned.
/// Thread-safe.
public final class GpuQueryStats {
  private final String _backendName;
  private final int _numSegments;
  private final AtomicInteger _segmentsOnGpu = new AtomicInteger();
  private final AtomicInteger _segmentsNoMatchingDocs = new AtomicInteger();
  private final AtomicInteger _segmentsIneligible = new AtomicInteger();
  private final AtomicInteger _segmentsSelectiveOnCpu = new AtomicInteger();
  private final AtomicInteger _segmentsNotResident = new AtomicInteger();
  private final AtomicInteger _segmentsFallback = new AtomicInteger();
  private final AtomicLong _docsOnGpu = new AtomicLong();
  private final AtomicLong _deviceTimeNs = new AtomicLong();

  public GpuQueryStats(String backendName, int numSegments) {
    _backendName = backendName;
    _numSegments = numSegments;
  }

  void recordGpuExecution(long numDocs, long deviceTimeNs) {
    _segmentsOnGpu.incrementAndGet();
    _docsOnGpu.addAndGet(numDocs);
    if (deviceTimeNs > 0) {
      _deviceTimeNs.addAndGet(deviceTimeNs);
    }
  }

  void recordNoMatchingDocs() {
    _segmentsNoMatchingDocs.incrementAndGet();
  }

  /// The segment used the regular CPU plan because it is not eligible (e.g. consuming, star-tree).
  void recordIneligible() {
    _segmentsIneligible.incrementAndGet();
  }

  /// The filter matched too few documents for the GPU to pay off; the segment ran on the CPU.
  void recordSelectiveOnCpu() {
    _segmentsSelectiveOnCpu.incrementAndGet();
  }

  /// The column could not be resident within the GPU memory budget; the segment ran on the CPU.
  void recordNotResident() {
    _segmentsNotResident.incrementAndGet();
  }

  void recordFallback() {
    _segmentsFallback.incrementAndGet();
  }

  public int getSegmentsOnGpu() {
    return _segmentsOnGpu.get();
  }

  public int getSegmentsSelectiveOnCpu() {
    return _segmentsSelectiveOnCpu.get();
  }

  public int getSegmentsNotResident() {
    return _segmentsNotResident.get();
  }

  public int getSegmentsFallback() {
    return _segmentsFallback.get();
  }

  public long getDeviceTimeNs() {
    return _deviceTimeNs.get();
  }

  @Override
  public String toString() {
    return String.format(Locale.ROOT,
        "backend=%s, segments=%d, segmentsOnGpu=%d, segmentsNoMatchingDocs=%d, segmentsIneligible=%d, "
            + "segmentsSelectiveOnCpu=%d, segmentsNotResident=%d, segmentsFallbackToCpu=%d, docsOnGpu=%d, "
            + "gpuKernelMs=%.3f", _backendName, _numSegments, _segmentsOnGpu.get(), _segmentsNoMatchingDocs.get(),
        _segmentsIneligible.get(), _segmentsSelectiveOnCpu.get(), _segmentsNotResident.get(),
        _segmentsFallback.get(), _docsOnGpu.get(), _deviceTimeNs.get() / 1e6);
  }
}

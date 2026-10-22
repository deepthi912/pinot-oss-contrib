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


/// A GPU execution backend (CUDA, Metal) for the per-row part of server-side aggregation queries. thread-safe: one
/// instance serves all concurrent queries of a server.
public interface GpuBackend extends AutoCloseable {

  GpuBackendType getType();

  String getName();

  /// Fills `buffer[0, length)` with the values of rows `[start, start + length)`.
  interface IntChunkReader {
    void read(int start, int length, int[] buffer);
  }

  /// Loads `numValues` values chunk by chunk into device memory, without materializing the whole column on the heap.
  /// The returned column may only be passed back to this backend.
  GpuColumn uploadIntColumn(int numValues, IntChunkReader reader);

  default GpuColumn uploadIntColumn(int[] values, int length) {
    return uploadIntColumn(length, (start, chunk, buffer) -> System.arraycopy(values, start, buffer, 0, chunk));
  }

  /// Uploads a per-query filter bitmap (bit `docId & 31` of word `docId >>> 5` per matching document).
  default GpuColumn uploadFilter(int[] words) {
    return uploadIntColumn(words, words.length);
  }

  /// Counts matching rows per group and, when `values` is set, sums its values exactly as 64-bit integers.
  /// `groupKeys` holds one group id in `[0, numGroups)` per document, or is `null` for a single group.
  /// `matchingDocs` comes from [#uploadFilter], so one upload serves every call for a segment; `null` means all
  /// documents match.
  GpuAggregationResult aggregate(@Nullable GpuColumn groupKeys, int numGroups, @Nullable GpuColumn values, int numDocs,
      @Nullable GpuColumn matchingDocs);

  /// Marks the distinct (group, id) pairs among matching documents; `ids` are in `[0, cardinality)`.
  GpuDistinctResult distinct(@Nullable GpuColumn groupKeys, int numGroups, GpuColumn ids, int cardinality,
      int numDocs, @Nullable GpuColumn matchingDocs);

  @Override
  void close();
}

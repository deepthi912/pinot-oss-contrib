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

  /// Copies `values[0, length)` to the device. The returned column may only be passed back to this backend.
  GpuColumn uploadIntColumn(int[] values, int length);

  /// Counts matching rows per dictionary id. `groupKeys` holds one id in `[0, cardinality)` per document;
  /// `matchingDocsWords` sets bit `docId & 31` of word `docId >>> 5` for each matching document, or is `null` when all
  /// match. Returns counts of length `cardinality`, indexed by dictionary id.
  GpuGroupByCountResult groupByCount(GpuColumn groupKeys, int cardinality, @Nullable int[] matchingDocsWords);

  @Override
  void close();
}

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


/// A GPU execution backend (CUDA, Metal) for the per-row part of server-side aggregation queries.
///
/// Pinot keeps filtering on the CPU, where its indexes live, and hands the backend the matching documents as a
/// bitmap; the backend runs the aggregation over device-resident columns. Operators are added one at a time, so this
/// interface grows with each supported operator.
///
/// Implementations must be thread-safe: one instance serves all concurrent queries of a server. A [GpuColumn] may only
/// be passed back to the backend that created it.
public interface GpuBackend extends AutoCloseable {

  GpuBackendType getType();

  /// Human-readable name including the device, e.g. `metal(Apple M2 Max)`.
  String getName();

  /// Copies `values[0, length)` into device memory and returns the resident column.
  GpuColumn uploadIntColumn(int[] values, int length);

  /// `SELECT groupKey, COUNT(*) ... GROUP BY groupKey` over dictionary ids.
  ///
  /// @param groupKeys dictionary ids of the group-by column, one per document, all in `[0, cardinality)`
  /// @param cardinality dictionary cardinality of the group-by column
  /// @param matchingDocsWords bitmap of matching documents, bit `d` of word `d >>> 5` set when document `d` matches;
  ///                          `null` when every document matches
  /// @return the number of matching documents per dictionary id (length `cardinality`) and the device time
  GpuGroupByCountResult groupByCount(GpuColumn groupKeys, int cardinality, @Nullable int[] matchingDocsWords);

  @Override
  void close();
}

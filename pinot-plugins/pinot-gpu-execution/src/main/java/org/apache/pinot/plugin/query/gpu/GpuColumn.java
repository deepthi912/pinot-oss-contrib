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

/// A column resident in a backend's device memory (or registered with it, for unified-memory devices).
///
/// Created by [GpuBackend#uploadIntColumn] and owned by the [GpuColumnCache]; [#close()] releases the device memory
/// and must be called exactly once, after the last query using the column has finished.
public interface GpuColumn extends AutoCloseable {

  /// Number of values (one per document of the segment).
  int getNumValues();

  /// Device memory held by this column, used for the cache's byte budget.
  long getSizeInBytes();

  @Override
  void close();
}

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

/// Discovers and creates one [GpuBackend], loaded with [java.util.ServiceLoader] from the plugin's class path.
/// Contract: [#getType()] is never CPU and matches the created backend; only the first provider per type is used.
public interface GpuBackendProvider {

  GpuBackendType getType();

  Availability probe();

  GpuBackend create();

  final class Availability {
    private final boolean _available;
    private final String _detail;

    private Availability(boolean available, String detail) {
      _available = available;
      _detail = detail;
    }

    public static Availability available(String detail) {
      return new Availability(true, detail);
    }

    public static Availability unavailable(String detail) {
      return new Availability(false, detail);
    }

    public boolean isAvailable() {
      return _available;
    }

    public String getDetail() {
      return _detail;
    }

    @Override
    public String toString() {
      return (_available ? "available: " : "unavailable: ") + _detail;
    }
  }
}

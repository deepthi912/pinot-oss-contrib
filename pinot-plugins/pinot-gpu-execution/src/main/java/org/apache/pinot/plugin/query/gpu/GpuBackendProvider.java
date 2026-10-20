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

/// Discovers and creates one [GpuBackend]. Providers are loaded with [java.util.ServiceLoader] through this plugin's
/// class loader, so a provider must be on the same class path as the plugin (packaged in its jar, or next to it on
/// the server class path); discovery across isolated plugin realms is not supported yet.
///
/// Contract: [#getType()] is never [GpuBackendType#CPU] (CPU is the built-in last resort and such providers are
/// ignored), the created backend reports the same type, and only the first provider of each type is used.
public interface GpuBackendProvider {

  GpuBackendType getType();

  /// Checks whether the hardware, driver and native library are present. Called once at server startup.
  Availability probe();

  /// Creates the backend. Only called after a successful [#probe()].
  GpuBackend create();

  /// Result of [#probe()]: whether the backend can be used here, and why (shown in logs and query decisions).
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

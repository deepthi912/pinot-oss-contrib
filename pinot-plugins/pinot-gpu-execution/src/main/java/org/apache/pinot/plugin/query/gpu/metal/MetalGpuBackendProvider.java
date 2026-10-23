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
package org.apache.pinot.plugin.query.gpu.metal;

import org.apache.pinot.plugin.query.gpu.GpuBackend;
import org.apache.pinot.plugin.query.gpu.GpuBackendProvider;
import org.apache.pinot.plugin.query.gpu.GpuBackendType;


/// Experimental Apple GPU backend: available on macOS / Apple Silicon when the native library is present. Stateless.
public class MetalGpuBackendProvider implements GpuBackendProvider {

  @Override
  public GpuBackendType getType() {
    return GpuBackendType.METAL;
  }

  @Override
  public Availability probe() {
    if (!MetalNativeLibrary.isAppleSilicon()) {
      return Availability.unavailable("not macOS on Apple Silicon");
    }
    String source = MetalNativeLibrary.describeSource();
    if (source == null) {
      return Availability.unavailable("native library not found (build the plugin with -Dpinot.gpu.metal=true, or "
          + "set -D" + MetalNativeLibrary.LIBRARY_PATH_PROPERTY + " to an existing library)");
    }
    return Availability.available(source);
  }

  @Override
  public GpuBackend create() {
    return new MetalGpuBackend(MetalNativeLibrary.resolve());
  }
}

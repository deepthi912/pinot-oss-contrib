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

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;
import javax.annotation.Nullable;


/// Locates the Metal backend's native library: the path in system property [#LIBRARY_PATH_PROPERTY] when set, otherwise
/// the copy packaged in the plugin jar (built on macOS by the `gpu-metal-native` Maven profile), extracted to a
/// temporary file. Thread-safe.
public final class MetalNativeLibrary {
  public static final String LIBRARY_PATH_PROPERTY = "pinot.gpu.metal.library";
  static final String RESOURCE = "/native/darwin-aarch64/libpinot_gpu_metal.dylib";

  private static volatile Path _extracted;

  private MetalNativeLibrary() {
  }

  public static boolean isAppleSilicon() {
    return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("mac") && "aarch64".equals(
        System.getProperty("os.arch"));
  }

  @Nullable
  public static String describeSource() {
    String configured = System.getProperty(LIBRARY_PATH_PROPERTY);
    if (configured != null) {
      return Files.exists(Path.of(configured)) ? configured : null;
    }
    return MetalNativeLibrary.class.getResource(RESOURCE) != null ? "classpath:" + RESOURCE : null;
  }

  public static synchronized Path resolve() {
    String configured = System.getProperty(LIBRARY_PATH_PROPERTY);
    if (configured != null) {
      return Path.of(configured);
    }
    if (_extracted != null) {
      return _extracted;
    }
    try (InputStream in = MetalNativeLibrary.class.getResourceAsStream(RESOURCE)) {
      if (in == null) {
        throw new IllegalStateException("Metal native library not packaged at " + RESOURCE);
      }
      Path file = Files.createTempFile("libpinot_gpu_metal", ".dylib");
      file.toFile().deleteOnExit();
      Files.copy(in, file, StandardCopyOption.REPLACE_EXISTING);
      _extracted = file;
      return file;
    } catch (IOException e) {
      throw new UncheckedIOException("Failed to extract the Metal native library", e);
    }
  }
}

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
package org.apache.pinot.integration.tests.gpu;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
import org.apache.pinot.plugin.query.gpu.GpuBackend;
import org.apache.pinot.plugin.query.gpu.GpuBackendProvider;
import org.apache.pinot.plugin.query.gpu.GpuBackendType;
import org.apache.pinot.plugin.query.gpu.GpuColumn;
import org.apache.pinot.plugin.query.gpu.GpuGroupByCountResult;


/// Plain-Java stand-in GPU backend discovered through ServiceLoader; available only while system property
/// [#ENABLED_PROPERTY] is true, so it does not affect other tests. Thread-safe.
public class TestGpuBackendProvider implements GpuBackendProvider {
  public static final String ENABLED_PROPERTY = "pinot.gpu.test.backend.enabled";
  private static final AtomicInteger NUM_GROUP_BY_CALLS = new AtomicInteger();

  public static int getNumGroupByCalls() {
    return NUM_GROUP_BY_CALLS.get();
  }

  @Override
  public GpuBackendType getType() {
    return GpuBackendType.CUDA;
  }

  @Override
  public Availability probe() {
    return Boolean.getBoolean(ENABLED_PROPERTY) ? Availability.available("integration test backend")
        : Availability.unavailable("integration test backend disabled");
  }

  @Override
  public GpuBackend create() {
    return new GpuBackend() {
      @Override
      public GpuBackendType getType() {
        return GpuBackendType.CUDA;
      }

      @Override
      public String getName() {
        return "test-gpu";
      }

      @Override
      public GpuColumn uploadIntColumn(int[] values, int length) {
        return new TestColumn(Arrays.copyOf(values, length));
      }

      @Override
      public GpuGroupByCountResult groupByCount(GpuColumn groupKeys, int cardinality,
          @Nullable int[] matchingDocsWords) {
        NUM_GROUP_BY_CALLS.incrementAndGet();
        int[] keys = ((TestColumn) groupKeys)._values;
        long[] counts = new long[cardinality];
        for (int docId = 0; docId < keys.length; docId++) {
          if (matchingDocsWords == null || (matchingDocsWords[docId >>> 5] >>> (docId & 31) & 1) != 0) {
            counts[keys[docId]]++;
          }
        }
        return new GpuGroupByCountResult(counts, -1);
      }

      @Override
      public void close() {
      }
    };
  }

  private static final class TestColumn implements GpuColumn {
    final int[] _values;

    TestColumn(int[] values) {
      _values = values;
    }

    @Override
    public int getNumValues() {
      return _values.length;
    }

    @Override
    public long getSizeInBytes() {
      return (long) _values.length * Integer.BYTES;
    }

    @Override
    public void close() {
    }
  }
}

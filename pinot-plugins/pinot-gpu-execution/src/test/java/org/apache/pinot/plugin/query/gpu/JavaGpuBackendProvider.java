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

import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;


/// Test stand-in for a GPU backend: implements the backend contract in plain Java and registers under any
/// [GpuBackendType] (it is not a CUDA implementation). It lets tests drive the whole GPU code path (plan maker,
/// operator, column cache, result conversion) on machines without a GPU, counts calls so tests can assert which path a
/// query took, can fail or return corrupt counts on demand, and rejects use of a column after it was freed.
/// Thread-safe.
public class JavaGpuBackendProvider implements GpuBackendProvider {

  public enum Behavior {
    CORRECT,
    FAIL,
    // Counts that don't add up to the filtered documents.
    CORRUPT,
    // Correct counts, but no sums and empty distinct bitmaps: reaches the checks after the count validation.
    CORRUPT_VALUES
  }

  private final GpuBackendType _type;
  private final boolean _available;
  private final Behavior _behavior;
  private final AtomicInteger _numGroupByCalls = new AtomicInteger();
  private final AtomicInteger _numUploads = new AtomicInteger();
  private final AtomicInteger _numBackendCloses = new AtomicInteger();

  public JavaGpuBackendProvider(GpuBackendType type, boolean available, Behavior behavior) {
    _type = type;
    _available = available;
    _behavior = behavior;
  }

  public int getNumGroupByCalls() {
    return _numGroupByCalls.get();
  }

  public int getNumUploads() {
    return _numUploads.get();
  }

  public int getNumBackendCloses() {
    return _numBackendCloses.get();
  }

  @Override
  public GpuBackendType getType() {
    return _type;
  }

  @Override
  public Availability probe() {
    return _available ? Availability.available("java test backend") : Availability.unavailable("test device missing");
  }

  @Override
  public GpuBackend create() {
    return new GpuBackend() {
      @Override
      public GpuBackendType getType() {
        return _type;
      }

      @Override
      public String getName() {
        return "java-" + _type.getId();
      }

      @Override
      public GpuColumn uploadIntColumn(int numValues, IntChunkReader reader) {
        _numUploads.incrementAndGet();
        int[] values = new int[numValues];
        // Odd chunk size above the reader's internal chunk so tests cover partial and multi-read chunks.
        int[] buffer = new int[12_345];
        for (int start = 0; start < numValues; start += buffer.length) {
          int length = Math.min(buffer.length, numValues - start);
          reader.read(start, length, buffer);
          System.arraycopy(buffer, 0, values, start, length);
        }
        return new JavaColumn(values);
      }

      @Override
      public GpuColumn uploadFilter(int[] words) {
        return new JavaColumn(words.clone());
      }

      @Override
      public GpuAggregationResult aggregate(@Nullable GpuColumn groupKeys, int numGroups, @Nullable GpuColumn values,
          int numDocs, @Nullable GpuColumn matchingDocs) {
        _numGroupByCalls.incrementAndGet();
        if (_behavior == Behavior.FAIL) {
          throw new IllegalStateException("simulated device failure");
        }
        int[] matchingDocsWords = matchingDocs != null ? ((JavaColumn) matchingDocs).values() : null;
        int[] keys = groupKeys != null ? ((JavaColumn) groupKeys).values() : null;
        int[] vals = values != null ? ((JavaColumn) values).values() : null;
        long[] counts = new long[numGroups];
        long[] sums = vals != null ? new long[numGroups] : null;
        for (int docId = 0; docId < numDocs; docId++) {
          if (matches(matchingDocsWords, docId)) {
            int g = keys != null ? keys[docId] : 0;
            counts[g]++;
            if (sums != null) {
              sums[g] += vals[docId];
            }
          }
        }
        if (_behavior == Behavior.CORRUPT) {
          counts[0]++;
        }
        return new GpuAggregationResult(counts, _behavior == Behavior.CORRUPT_VALUES ? null : sums, -1);
      }

      @Override
      public GpuDistinctResult distinct(@Nullable GpuColumn groupKeys, int numGroups, GpuColumn ids,
          int cardinality, int numDocs, @Nullable GpuColumn matchingDocs) {
        _numGroupByCalls.incrementAndGet();
        if (_behavior == Behavior.FAIL) {
          throw new IllegalStateException("simulated device failure");
        }
        int[] matchingDocsWords = matchingDocs != null ? ((JavaColumn) matchingDocs).values() : null;
        int[] keys = groupKeys != null ? ((JavaColumn) groupKeys).values() : null;
        int[] idValues = ((JavaColumn) ids).values();
        int[] words = new int[(int) (((long) numGroups * cardinality + 31) >>> 5)];
        for (int docId = 0; docId < numDocs; docId++) {
          if (matches(matchingDocsWords, docId)) {
            long bit = (long) (keys != null ? keys[docId] : 0) * cardinality + idValues[docId];
            words[(int) (bit >>> 5)] |= 1 << (bit & 31);
          }
        }
        return new GpuDistinctResult(_behavior == Behavior.CORRUPT_VALUES ? new int[words.length] : words, -1);
      }

      @Override
      public void close() {
        _numBackendCloses.incrementAndGet();
      }
    };
  }

  private static boolean matches(@Nullable int[] words, int docId) {
    return words == null || (words[docId >>> 5] >>> (docId & 31) & 1) != 0;
  }

  static final class JavaColumn implements GpuColumn {
    private final int[] _values;
    private volatile boolean _closed;

    JavaColumn(int[] values) {
      _values = values;
    }

    int[] values() {
      if (_closed) {
        throw new AssertionError("column used after it was freed");
      }
      return _values;
    }

    boolean isClosed() {
      return _closed;
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
    public synchronized void close() {
      if (_closed) {
        throw new AssertionError("column freed twice");
      }
      _closed = true;
    }
  }
}

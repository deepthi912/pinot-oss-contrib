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

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import javax.annotation.Nullable;
import org.apache.pinot.plugin.query.gpu.GpuAggregationResult;
import org.apache.pinot.plugin.query.gpu.GpuBackend;
import org.apache.pinot.plugin.query.gpu.GpuBackendType;
import org.apache.pinot.plugin.query.gpu.GpuColumn;
import org.apache.pinot.plugin.query.gpu.GpuDistinctResult;


/// Experimental Apple GPU backend on Metal, called through the FFM API (native code: `src/main/native/metal_ops.mm`).
/// Thread-safe: Metal command queues accept concurrent submissions, kernels only read the shared resident columns and
/// each call allocates its own output buffers. Native calls hold the read lock and [#close()] the write lock, so the
/// library is never unloaded under a running call; after close every call throws [IllegalStateException] (callers
/// recompute on the CPU).
public class MetalGpuBackend implements GpuBackend {
  private static final long PAGE_SIZE = 16 * 1024;
  private static final int ERROR_LENGTH = 1024;
  private static final int UPLOAD_CHUNK_SIZE = 64 * 1024;
  private static final AddressLayout ADDRESS = ValueLayout.ADDRESS;
  private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT;
  private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG;

  private final Arena _libraryArena = Arena.ofShared();
  private final MethodHandle _destroy;
  private final MethodHandle _pin;
  private final MethodHandle _unpin;
  private final MethodHandle _groupBy;
  private final MethodHandle _distinct;
  private final MemorySegment _context;
  private final String _deviceName;
  private final ReentrantReadWriteLock _lock = new ReentrantReadWriteLock();
  // Guarded by _lock.
  private boolean _closed;

  public MetalGpuBackend(Path library) {
    Linker linker = Linker.nativeLinker();
    SymbolLookup lookup = SymbolLookup.libraryLookup(library, _libraryArena);
    MethodHandle create = linker.downcallHandle(lookup.find("mg_create").orElseThrow(),
        FunctionDescriptor.of(ADDRESS, ADDRESS, INT));
    MethodHandle deviceName = linker.downcallHandle(lookup.find("mg_device_name").orElseThrow(),
        FunctionDescriptor.of(ADDRESS, ADDRESS));
    _destroy = linker.downcallHandle(lookup.find("mg_destroy").orElseThrow(), FunctionDescriptor.ofVoid(ADDRESS));
    _pin = linker.downcallHandle(lookup.find("mg_pin").orElseThrow(),
        FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, LONG));
    _unpin = linker.downcallHandle(lookup.find("mg_unpin").orElseThrow(), FunctionDescriptor.ofVoid(ADDRESS));
    _groupBy = linker.downcallHandle(lookup.find("mg_group_by").orElseThrow(),
        FunctionDescriptor.of(INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, INT, INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS,
            INT));
    _distinct = linker.downcallHandle(lookup.find("mg_distinct_pairs").orElseThrow(),
        FunctionDescriptor.of(INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, INT, INT, INT, ADDRESS, ADDRESS, ADDRESS, INT));
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment error = arena.allocate(ERROR_LENGTH);
      _context = (MemorySegment) create.invokeExact(error, ERROR_LENGTH);
      if (_context.equals(MemorySegment.NULL)) {
        throw new IllegalStateException("Failed to initialize Metal: " + error.getString(0));
      }
      _deviceName = ((MemorySegment) deviceName.invokeExact(_context)).reinterpret(256).getString(0);
    } catch (RuntimeException | Error e) {
      _libraryArena.close();
      throw e;
    } catch (Throwable t) {
      _libraryArena.close();
      throw new IllegalStateException("Failed to initialize Metal", t);
    }
  }

  @Override
  public GpuBackendType getType() {
    return GpuBackendType.METAL;
  }

  @Override
  public String getName() {
    return "metal(" + _deviceName + ")";
  }

  @Override
  public GpuColumn uploadIntColumn(int numValues, IntChunkReader reader) {
    Arena arena = Arena.ofShared();
    try {
      MemorySegment segment = allocateAligned(arena, numValues);
      int[] buffer = new int[Math.min(UPLOAD_CHUNK_SIZE, Math.max(numValues, 1))];
      for (int start = 0; start < numValues; start += buffer.length) {
        int length = Math.min(buffer.length, numValues - start);
        reader.read(start, length, buffer);
        MemorySegment.copy(buffer, 0, segment, ValueLayout.JAVA_INT, (long) start * Integer.BYTES, length);
      }
      return newColumn(arena, segment, numValues);
    } catch (RuntimeException | Error e) {
      arena.close();
      throw e;
    }
  }

  @Override
  public GpuColumn uploadFilter(int[] words) {
    // Per-segment and closed by the uploading thread: a confined arena avoids the global handshake of shared ones.
    Arena arena = Arena.ofConfined();
    try {
      MemorySegment segment = allocateAligned(arena, words.length);
      MemorySegment.copy(words, 0, segment, ValueLayout.JAVA_INT, 0, words.length);
      return newColumn(arena, segment, words.length);
    } catch (RuntimeException | Error e) {
      arena.close();
      throw e;
    }
  }

  @Override
  public GpuAggregationResult aggregate(@Nullable GpuColumn groupKeys, int numGroups, @Nullable GpuColumn values,
      int numDocs, @Nullable GpuColumn matchingDocs) {
    Lock lock = lockOpen();
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment counts = arena.allocate(ValueLayout.JAVA_LONG, numGroups);
      MemorySegment sumsU = values != null ? arena.allocate(ValueLayout.JAVA_LONG, numGroups) : MemorySegment.NULL;
      MemorySegment gpuMillis = arena.allocate(ValueLayout.JAVA_DOUBLE);
      MemorySegment error = arena.allocate(ERROR_LENGTH);
      int status = (int) _groupBy.invokeExact(_context, pinOf(groupKeys), pinOf(values), pinOf(matchingDocs),
          numDocs, numGroups, sumsU, counts, gpuMillis, error, ERROR_LENGTH);
      if (status != 0) {
        throw new IllegalStateException("Metal aggregation failed: " + error.getString(0));
      }
      long[] countArray = counts.toArray(ValueLayout.JAVA_LONG);
      long[] sums = null;
      if (values != null) {
        // The kernel sums (value + 2^31) to stay unsigned; subtract the bias per row.
        sums = sumsU.toArray(ValueLayout.JAVA_LONG);
        for (int g = 0; g < numGroups; g++) {
          sums[g] -= countArray[g] << 31;
        }
      }
      return new GpuAggregationResult(countArray, sums, deviceTimeNs(gpuMillis));
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new IllegalStateException("Metal aggregation failed", t);
    } finally {
      lock.unlock();
    }
  }

  @Override
  public GpuDistinctResult distinct(@Nullable GpuColumn groupKeys, int numGroups, GpuColumn ids, int cardinality,
      int numDocs, @Nullable GpuColumn matchingDocs) {
    Lock lock = lockOpen();
    try (Arena arena = Arena.ofConfined()) {
      int numWords = (int) (((long) numGroups * cardinality + 31) >>> 5);
      // The kernel writes the bitmap straight into this page-aligned memory: no device-side copy.
      MemorySegment words = allocateAligned(arena, numWords);
      MemorySegment wordsPin = pin(words);
      try {
        MemorySegment gpuMillis = arena.allocate(ValueLayout.JAVA_DOUBLE);
        MemorySegment error = arena.allocate(ERROR_LENGTH);
        int status = (int) _distinct.invokeExact(_context, pinOf(groupKeys), pinOf(ids), pinOf(matchingDocs),
            numDocs, numGroups, cardinality, wordsPin, gpuMillis, error, ERROR_LENGTH);
        if (status != 0) {
          throw new IllegalStateException("Metal distinct failed: " + error.getString(0));
        }
        return new GpuDistinctResult(words.asSlice(0, (long) numWords * Integer.BYTES).toArray(ValueLayout.JAVA_INT),
            deviceTimeNs(gpuMillis));
      } finally {
        unpin(wordsPin);
      }
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new IllegalStateException("Metal distinct failed", t);
    } finally {
      lock.unlock();
    }
  }

  /// Takes the read lock, or throws when the backend is closed.
  private Lock lockOpen() {
    Lock lock = _lock.readLock();
    lock.lock();
    if (_closed) {
      lock.unlock();
      throw new IllegalStateException("Metal backend is closed");
    }
    return lock;
  }

  private MetalColumn newColumn(Arena arena, MemorySegment segment, int numValues) {
    Lock lock = lockOpen();
    try {
      return new MetalColumn(arena, pin(segment), numValues, segment.byteSize());
    } finally {
      lock.unlock();
    }
  }

  private static MemorySegment pinOf(@Nullable GpuColumn column) {
    return column != null ? ((MetalColumn) column)._pin : MemorySegment.NULL;
  }

  private static long deviceTimeNs(MemorySegment gpuMillis) {
    return Math.round(gpuMillis.get(ValueLayout.JAVA_DOUBLE, 0) * 1_000_000);
  }

  private static MemorySegment allocateAligned(Arena arena, int numInts) {
    long bytes = (long) numInts * Integer.BYTES;
    return arena.allocate(Math.max(PAGE_SIZE, (bytes + PAGE_SIZE - 1) / PAGE_SIZE * PAGE_SIZE), PAGE_SIZE);
  }


  private MemorySegment pin(MemorySegment segment) {
    try {
      MemorySegment handle = (MemorySegment) _pin.invokeExact(_context, segment, segment.byteSize());
      if (handle.equals(MemorySegment.NULL)) {
        throw new IllegalStateException("Metal could not register " + segment.byteSize() + " bytes");
      }
      return handle;
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new IllegalStateException("Metal pin failed", t);
    }
  }

  private void unpin(MemorySegment handle) {
    if (handle.equals(MemorySegment.NULL)) {
      return;
    }
    try {
      _unpin.invokeExact(handle);
    } catch (Throwable t) {
      throw new IllegalStateException("Metal unpin failed", t);
    }
  }

  @Override
  public void close() {
    _lock.writeLock().lock();
    try {
      if (_closed) {
        return;
      }
      _closed = true;
      try {
        _destroy.invokeExact(_context);
      } catch (Throwable t) {
        throw new IllegalStateException("Failed to close Metal", t);
      } finally {
        _libraryArena.close();
      }
    } finally {
      _lock.writeLock().unlock();
    }
  }

  private final class MetalColumn implements GpuColumn {
    private final Arena _arena;
    private final MemorySegment _pin;
    private final int _numValues;
    private final long _sizeInBytes;
    private boolean _closed;

    MetalColumn(Arena arena, MemorySegment pin, int numValues, long sizeInBytes) {
      _arena = arena;
      _pin = pin;
      _numValues = numValues;
      _sizeInBytes = sizeInBytes;
    }

    @Override
    public int getNumValues() {
      return _numValues;
    }

    @Override
    public long getSizeInBytes() {
      return _sizeInBytes;
    }

    @Override
    public synchronized void close() {
      if (_closed) {
        return;
      }
      _closed = true;
      Lock lock = _lock.readLock();
      lock.lock();
      try {
        // Once the backend is closed the library is unloaded, so the native buffer handle is leaked, not released.
        if (!MetalGpuBackend.this._closed) {
          unpin(_pin);
        }
      } finally {
        lock.unlock();
        _arena.close();
      }
    }
  }
}

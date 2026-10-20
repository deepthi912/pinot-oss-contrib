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
import javax.annotation.Nullable;
import org.apache.pinot.plugin.query.gpu.GpuBackend;
import org.apache.pinot.plugin.query.gpu.GpuBackendType;
import org.apache.pinot.plugin.query.gpu.GpuColumn;
import org.apache.pinot.plugin.query.gpu.GpuGroupByCountResult;


/// Experimental Apple GPU backend on Metal, called through the FFM API (native code: `src/main/native/metal_ops.mm`).
///
/// Apple Silicon has unified memory, so a resident column is page-aligned off-heap memory that Metal wraps without
/// copying; registering it once and reusing it across queries avoids the per-call cost of mapping large buffers.
///
/// Thread-safe: Metal command queues accept concurrent submissions, and each call uses its own native buffers.
public class MetalGpuBackend implements GpuBackend {
  /// Apple Silicon page size; Metal no-copy buffers need page-aligned address and length.
  private static final long PAGE_SIZE = 16 * 1024;
  private static final int ERROR_LENGTH = 1024;
  private static final AddressLayout ADDRESS = ValueLayout.ADDRESS;
  private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT;
  private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG;

  private final Arena _libraryArena = Arena.ofShared();
  private final MethodHandle _destroy;
  private final MethodHandle _pin;
  private final MethodHandle _unpin;
  private final MethodHandle _groupBy;
  private final MemorySegment _context;
  private final String _deviceName;

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
  public GpuColumn uploadIntColumn(int[] values, int length) {
    Arena arena = Arena.ofShared();
    try {
      MemorySegment segment = copyAligned(arena, values, length);
      return new MetalColumn(arena, pin(segment), length, segment.byteSize());
    } catch (RuntimeException e) {
      arena.close();
      throw e;
    }
  }

  @Override
  public GpuGroupByCountResult groupByCount(GpuColumn groupKeys, int cardinality,
      @Nullable int[] matchingDocsWords) {
    MetalColumn keys = (MetalColumn) groupKeys;
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment filterPin = MemorySegment.NULL;
      if (matchingDocsWords != null) {
        filterPin = pin(copyAligned(arena, matchingDocsWords, matchingDocsWords.length));
      }
      try {
        MemorySegment counts = arena.allocate(ValueLayout.JAVA_LONG, cardinality);
        MemorySegment gpuMillis = arena.allocate(ValueLayout.JAVA_DOUBLE);
        MemorySegment error = arena.allocate(ERROR_LENGTH);
        int status = (int) _groupBy.invokeExact(_context, keys._pin, MemorySegment.NULL, filterPin,
            keys._numValues, cardinality, MemorySegment.NULL, counts, gpuMillis, error, ERROR_LENGTH);
        if (status != 0) {
          throw new IllegalStateException("Metal group-by failed: " + error.getString(0));
        }
        // GPU start/end timestamps of the command buffer, measured by the device itself.
        long deviceTimeNs = Math.round(gpuMillis.get(ValueLayout.JAVA_DOUBLE, 0) * 1_000_000);
        return new GpuGroupByCountResult(counts.toArray(ValueLayout.JAVA_LONG), deviceTimeNs);
      } finally {
        unpin(filterPin);
      }
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new IllegalStateException("Metal group-by failed", t);
    }
  }

  private static MemorySegment copyAligned(Arena arena, int[] values, int length) {
    long bytes = (long) length * Integer.BYTES;
    long padded = Math.max(PAGE_SIZE, (bytes + PAGE_SIZE - 1) / PAGE_SIZE * PAGE_SIZE);
    MemorySegment segment = arena.allocate(padded, PAGE_SIZE);
    MemorySegment.copy(values, 0, segment, ValueLayout.JAVA_INT, 0, length);
    return segment;
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
    try {
      _destroy.invokeExact(_context);
    } catch (Throwable t) {
      throw new IllegalStateException("Failed to close Metal", t);
    } finally {
      _libraryArena.close();
    }
  }

  /// Page-aligned off-heap copy of a column, registered with Metal once.
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
      try {
        unpin(_pin);
      } finally {
        _arena.close();
      }
    }
  }
}

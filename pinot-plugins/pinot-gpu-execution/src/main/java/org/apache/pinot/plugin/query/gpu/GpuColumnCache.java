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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import org.apache.pinot.segment.spi.IndexSegment;


/// Keeps segment columns resident in GPU memory across queries within a byte budget. Columns used within the
/// residency window are never evicted to make room, so a working set larger than the budget stays partly resident
/// (the rest runs on the CPU) instead of thrashing. Thread-safe; the monitor only guards bookkeeping.
///
/// TODO: release entries when a segment is unloaded; today they age out through eviction only.
public class GpuColumnCache implements AutoCloseable {
  public static final long DEFAULT_MIN_RESIDENCY_MS = 60_000;

  public final class Lease implements AutoCloseable {
    private final Entry _entry;
    private boolean _released;

    private Lease(Entry entry) {
      _entry = entry;
    }

    public GpuColumn getColumn() {
      return _entry._column;
    }

    @Override
    public void close() {
      boolean free;
      synchronized (GpuColumnCache.this) {
        if (_released) {
          return;
        }
        _released = true;
        _entry._refCount--;
        free = _entry.markClosedIfUnused();
        if (free) {
          _pendingFreeBytes -= _entry._sizeInBytes;
        }
      }
      if (free) {
        _entry._column.close();
      }
    }
  }

  private static final class Entry {
    final GpuColumn _column;
    final long _sizeInBytes;
    long _lastAccessNs;
    int _refCount;
    boolean _evicted;
    boolean _closed;

    Entry(GpuColumn column) {
      _column = column;
      _sizeInBytes = column.getSizeInBytes();
    }

    boolean markClosedIfUnused() {
      if (_evicted && _refCount == 0 && !_closed) {
        _closed = true;
        return true;
      }
      return false;
    }
  }

  private final Map<String, Entry> _entries = new LinkedHashMap<>(16, 0.75f, true);
  private final Map<String, CompletableFuture<Entry>> _loading = new HashMap<>();
  private final Map<IndexSegment, Long> _segmentInstanceIds = new WeakHashMap<>();
  private final long _maxBytes;
  private final long _minResidencyNs;
  private final LongSupplier _clockNs;
  private long _nextSegmentInstanceId;
  private long _residentBytes;
  private long _loadingBytes;
  private long _pendingFreeBytes;

  public GpuColumnCache(long maxBytes) {
    this(maxBytes, DEFAULT_MIN_RESIDENCY_MS, System::nanoTime);
  }

  public GpuColumnCache(long maxBytes, long minResidencyMs, LongSupplier clockNs) {
    _maxBytes = maxBytes;
    _minResidencyNs = TimeUnit.MILLISECONDS.toNanos(minResidencyMs);
    _clockNs = clockNs;
  }

  public static String cacheKey(GpuBackendType backend, String tableName, String segmentName,
      long segmentInstanceId, String column) {
    return backend.getId() + '|' + tableName + '|' + segmentName + '|' + segmentInstanceId + '|' + column;
  }

  // Per loaded instance: a reload can rebuild a column without changing segment name or CRC.
  public synchronized long getOrAssignInstanceId(IndexSegment segment) {
    return _segmentInstanceIds.computeIfAbsent(segment, k -> _nextSegmentInstanceId++);
  }

  /// Returns a lease on the resident column, loading it on a miss, or `null` when it cannot be made resident without
  /// evicting recently used columns (the caller then runs on the CPU).
  @Nullable
  public Lease acquire(String key, long estimatedBytes, Supplier<GpuColumn> loader) {
    while (true) {
      CompletableFuture<Entry> pending;
      CompletableFuture<Entry> ownLoad = null;
      List<GpuColumn> toFree = new ArrayList<>();
      boolean admitted = true;
      synchronized (this) {
        Entry entry = _entries.get(key);
        if (entry != null) {
          entry._refCount++;
          entry._lastAccessNs = _clockNs.getAsLong();
          return new Lease(entry);
        }
        pending = _loading.get(key);
        if (pending == null) {
          admitted = makeRoom(estimatedBytes, toFree);
          if (admitted) {
            _loadingBytes += estimatedBytes;
            ownLoad = new CompletableFuture<>();
            _loading.put(key, ownLoad);
            pending = ownLoad;
          }
        }
      }
      freeAll(toFree);
      if (!admitted) {
        return null;
      }
      if (ownLoad != null) {
        return load(key, estimatedBytes, loader, ownLoad);
      }
      Entry loaded;
      try {
        loaded = pending.join();
      } catch (CompletionException e) {
        throw e.getCause() instanceof RuntimeException ? (RuntimeException) e.getCause() : e;
      }
      synchronized (this) {
        if (!loaded._closed) {
          loaded._refCount++;
          loaded._lastAccessNs = _clockNs.getAsLong();
          return new Lease(loaded);
        }
      }
    }
  }

  private Lease load(String key, long estimatedBytes, Supplier<GpuColumn> loader, CompletableFuture<Entry> ownLoad) {
    Entry entry;
    try {
      entry = new Entry(loader.get());
    } catch (RuntimeException | Error e) {
      synchronized (this) {
        _loading.remove(key);
        _loadingBytes -= estimatedBytes;
      }
      ownLoad.completeExceptionally(e);
      throw e;
    }
    synchronized (this) {
      _loading.remove(key);
      _loadingBytes -= estimatedBytes;
      _entries.put(key, entry);
      _residentBytes += entry._sizeInBytes;
      entry._refCount++;
      entry._lastAccessNs = _clockNs.getAsLong();
    }
    ownLoad.complete(entry);
    return new Lease(entry);
  }

  /// Evicts least recently used columns idle for longer than the residency window until `neededBytes` fit.
  private boolean makeRoom(long neededBytes, List<GpuColumn> toFree) {
    if (neededBytes > _maxBytes) {
      return false;
    }
    long protectedSince = _clockNs.getAsLong() - _minResidencyNs;
    Iterator<Entry> iterator = _entries.values().iterator();
    while (_residentBytes + _loadingBytes + _pendingFreeBytes + neededBytes > _maxBytes) {
      if (!iterator.hasNext()) {
        return false;
      }
      Entry eldest = iterator.next();
      if (eldest._lastAccessNs > protectedSince) {
        return false;
      }
      iterator.remove();
      _residentBytes -= eldest._sizeInBytes;
      eldest._evicted = true;
      if (eldest.markClosedIfUnused()) {
        toFree.add(eldest._column);
      } else {
        _pendingFreeBytes += eldest._sizeInBytes;
      }
    }
    return true;
  }

  public synchronized long getResidentBytes() {
    return _residentBytes;
  }

  public synchronized int getNumResidentColumns() {
    return _entries.size();
  }

  private static void freeAll(List<GpuColumn> columns) {
    for (GpuColumn column : columns) {
      column.close();
    }
  }

  @Override
  public void close() {
    List<GpuColumn> toFree = new ArrayList<>();
    synchronized (this) {
      for (Entry entry : _entries.values()) {
        entry._evicted = true;
        if (entry.markClosedIfUnused()) {
          toFree.add(entry._column);
        } else {
          _pendingFreeBytes += entry._sizeInBytes;
        }
      }
      _entries.clear();
      _residentBytes = 0;
    }
    freeAll(toFree);
  }
}

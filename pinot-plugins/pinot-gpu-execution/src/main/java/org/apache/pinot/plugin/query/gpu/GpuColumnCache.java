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
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.LongSupplier;
import java.util.function.Predicate;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import org.apache.pinot.segment.spi.IndexSegment;


/// Keeps segment columns resident in GPU memory across queries within a byte budget. The budget counts off-heap native
/// memory (device memory, or host memory registered with a unified-memory GPU), not Java heap. Columns used within the
/// residency window are never evicted to make room, so a working set larger than the budget stays partly resident
/// (the rest runs on the CPU) instead of thrashing. Thread-safe; the monitor only guards bookkeeping.
///
/// Columns of segment instances that were released ([#releaseSegment]) or garbage collected (unloaded or reloaded
/// segments) are freed regardless of residency: eagerly on release, and on the next [#acquire] after a collection.
/// The plugin has no segment-unload hook yet, so servers rely on garbage collection.
public class GpuColumnCache implements AutoCloseable {
  public static final long DEFAULT_MIN_RESIDENCY_MS = 60_000;
  public static final int MAX_CONCURRENT_LOADS = 2;

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
    final long _segmentInstanceId;
    long _lastAccessNs;
    int _refCount;
    boolean _evicted;
    boolean _closed;

    Entry(GpuColumn column, long segmentInstanceId) {
      _column = column;
      _sizeInBytes = column.getSizeInBytes();
      _segmentInstanceId = segmentInstanceId;
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
  // Ids of the instances in _segmentInstanceIds; when the weak map holds fewer, some segment was collected.
  private final Set<Long> _liveInstanceIds = new HashSet<>();
  private final long _maxBytes;
  private final long _minResidencyNs;
  private final LongSupplier _clockNs;
  private long _nextSegmentInstanceId;
  private boolean _closed;
  private long _residentBytes;
  private long _loadingBytes;
  private long _pendingFreeBytes;
  private final Semaphore _loadPermits = new Semaphore(MAX_CONCURRENT_LOADS);

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

  /// Per loaded instance: a reload can rebuild a column without changing segment name or CRC.
  public synchronized long getOrAssignInstanceId(IndexSegment segment) {
    return _segmentInstanceIds.computeIfAbsent(segment, k -> {
      long id = _nextSegmentInstanceId++;
      _liveInstanceIds.add(id);
      return id;
    });
  }

  /// Returns a lease on the resident column, loading it on a miss, or `null` when it cannot be made resident without
  /// evicting recently used columns, or not within `maxWaitNanos` (the caller then runs on the CPU). At most
  /// [#MAX_CONCURRENT_LOADS] loads run at once; further misses wait for a slot, bounded by `maxWaitNanos`.
  /// `estimatedBytes` should match the loaded column's [GpuColumn#getSizeInBytes()].
  @Nullable
  public Lease acquire(String key, long segmentInstanceId, long estimatedBytes, long maxWaitNanos,
      Supplier<GpuColumn> loader) {
    long deadlineNanos = System.nanoTime() + maxWaitNanos;
    boolean holdsPermit = false;
    try {
      while (true) {
        CompletableFuture<Entry> pending;
        boolean ownLoad = false;
        boolean refused = false;
        Lease hit = null;
        List<GpuColumn> toFree = new ArrayList<>();
        synchronized (this) {
          if (_closed) {
            return null;
          }
          sweepCollectedSegments(toFree);
          Entry entry = _entries.get(key);
          pending = null;
          if (entry != null) {
            entry._refCount++;
            entry._lastAccessNs = _clockNs.getAsLong();
            hit = new Lease(entry);
          } else {
            pending = _loading.get(key);
            if (pending == null && holdsPermit) {
              if (makeRoom(estimatedBytes, toFree)) {
                _loadingBytes += estimatedBytes;
                pending = new CompletableFuture<>();
                _loading.put(key, pending);
                ownLoad = true;
              }
            } else if (pending == null && !canMakeRoom(estimatedBytes)) {
              refused = true;
            }
          }
        }
        freeAll(toFree);
        if (hit != null) {
          return hit;
        }
        if (refused) {
          return null;
        }
        if (ownLoad) {
          return load(key, segmentInstanceId, estimatedBytes, loader, pending);
        }
        if (pending == null) {
          if (holdsPermit || !_loadPermits.tryAcquire(Math.max(0, deadlineNanos - System.nanoTime()),
              TimeUnit.NANOSECONDS)) {
            return null;
          }
          holdsPermit = true;
          continue;
        }
        if (holdsPermit) {
          // Another thread loads this key: don't hold a slot while waiting for it.
          _loadPermits.release();
          holdsPermit = false;
        }
        Entry loaded;
        try {
          loaded = pending.get(Math.max(0, deadlineNanos - System.nanoTime()), TimeUnit.NANOSECONDS);
        } catch (TimeoutException e) {
          return null;
        } catch (ExecutionException e) {
          // The other load failed, possibly because its own query was cancelled: retry with this caller's loader.
          continue;
        }
        synchronized (this) {
          if (!loaded._closed) {
            loaded._refCount++;
            loaded._lastAccessNs = _clockNs.getAsLong();
            return new Lease(loaded);
          }
        }
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      return null;
    } finally {
      if (holdsPermit) {
        _loadPermits.release();
      }
    }
  }

  private Lease load(String key, long segmentInstanceId, long estimatedBytes, Supplier<GpuColumn> loader,
      CompletableFuture<Entry> ownLoad) {
    Entry entry;
    try {
      entry = new Entry(loader.get(), segmentInstanceId);
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
      entry._refCount++;
      entry._lastAccessNs = _clockNs.getAsLong();
      if (_closed || !_liveInstanceIds.contains(segmentInstanceId)) {
        // Released or closed while loading: not cached. This caller (and callers already waiting for this load)
        // get leases; the column is freed when the last one closes.
        entry._evicted = true;
        _pendingFreeBytes += entry._sizeInBytes;
      } else {
        _entries.put(key, entry);
        _residentBytes += entry._sizeInBytes;
      }
    }
    ownLoad.complete(entry);
    return new Lease(entry);
  }

  /// Frees the columns of segment instances that were garbage collected. `size()` drops collected keys, so the sweep
  /// over the entries only runs after an instance actually went away.
  private void sweepCollectedSegments(List<GpuColumn> toFree) {
    if (_segmentInstanceIds.size() < _liveInstanceIds.size()) {
      _liveInstanceIds.retainAll(new HashSet<>(_segmentInstanceIds.values()));
      evict(entry -> !_liveInstanceIds.contains(entry._segmentInstanceId), toFree);
    }
  }

  /// Whether [#makeRoom] could succeed, without evicting anything; checked before waiting for a load slot.
  private boolean canMakeRoom(long neededBytes) {
    if (neededBytes > _maxBytes) {
      return false;
    }
    long excess = _residentBytes + _loadingBytes + _pendingFreeBytes + neededBytes - _maxBytes;
    if (excess <= 0) {
      return true;
    }
    long protectedSince = _clockNs.getAsLong() - _minResidencyNs;
    for (Entry entry : _entries.values()) {
      if (entry._lastAccessNs > protectedSince) {
        return false;
      }
      excess -= entry._sizeInBytes;
      if (excess <= 0) {
        return true;
      }
    }
    return false;
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
      evicted(eldest, toFree);
    }
    return true;
  }

  private void evict(Predicate<Entry> condition, List<GpuColumn> toFree) {
    Iterator<Entry> iterator = _entries.values().iterator();
    while (iterator.hasNext()) {
      Entry entry = iterator.next();
      if (condition.test(entry)) {
        iterator.remove();
        evicted(entry, toFree);
      }
    }
  }

  private void evicted(Entry entry, List<GpuColumn> toFree) {
    _residentBytes -= entry._sizeInBytes;
    entry._evicted = true;
    if (entry.markClosedIfUnused()) {
      toFree.add(entry._column);
    } else {
      _pendingFreeBytes += entry._sizeInBytes;
    }
  }

  /// Frees the columns of a segment instance, e.g. when the segment is unloaded.
  public void releaseSegment(IndexSegment segment) {
    List<GpuColumn> toFree = new ArrayList<>();
    synchronized (this) {
      Long instanceId = _segmentInstanceIds.remove(segment);
      if (instanceId != null) {
        _liveInstanceIds.remove(instanceId);
        evict(entry -> entry._segmentInstanceId == instanceId, toFree);
      }
    }
    freeAll(toFree);
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

  /// Frees idle columns now and leased ones when their leases close; later acquires return `null`. Idempotent.
  @Override
  public void close() {
    List<GpuColumn> toFree = new ArrayList<>();
    synchronized (this) {
      if (_closed) {
        return;
      }
      _closed = true;
      evict(entry -> true, toFree);
    }
    freeAll(toFree);
  }
}

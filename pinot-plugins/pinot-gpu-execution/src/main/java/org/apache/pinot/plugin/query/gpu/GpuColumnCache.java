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
import java.util.function.Supplier;
import javax.annotation.Nullable;
import org.apache.pinot.segment.spi.IndexSegment;


/// Keeps segment columns resident in GPU memory across queries, bounded by a byte budget with LRU eviction.
///
/// Residency is what makes GPU execution pay off: copying a column to the device on every query costs more than the
/// aggregation saves. Callers key entries per loaded segment instance (see [#key]), so a reloaded or replaced segment
/// never reuses a stale column.
///
/// Thread-safety: the cache monitor only guards bookkeeping (entries, reference counts, LRU order, resident bytes).
/// Loading a column (reading the forward index, copying it to the device) and freeing evicted columns happen outside
/// the monitor. Concurrent misses on the same key share one load. A column is reference-counted while queries use it
/// ([Lease]); an evicted column is freed when its last lease is released.
///
/// TODO: release entries when a segment is unloaded; today stale entries age out through LRU eviction only.
public class GpuColumnCache implements AutoCloseable {

  /// A column in use by one query; [#close()] releases it.
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
      }
      if (free) {
        _entry._column.close();
      }
    }
  }

  private static final class Entry {
    final GpuColumn _column;
    int _refCount;
    boolean _evicted;
    boolean _closed;

    Entry(GpuColumn column) {
      _column = column;
    }

    /// Called under the cache monitor; returns true when the caller must free the column (outside the monitor).
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
  /// Weak, identity-based: IndexSegment does not override equals/hashCode, and an unloaded segment's id goes away.
  private final Map<IndexSegment, Long> _segmentInstanceIds = new WeakHashMap<>();
  private long _nextSegmentInstanceId;
  private long _maxBytes;
  private long _residentBytes;

  public GpuColumnCache(long maxBytes) {
    _maxBytes = maxBytes;
  }

  /// Cache key for one column of one loaded segment instance. `segmentInstanceId` must change whenever the segment is
  /// (re)loaded, e.g. the identity of the `IndexSegment` object, because a reload can rebuild a column in place
  /// without changing the segment name or CRC.
  public static String key(GpuBackendType backend, String tableName, String segmentName, long segmentInstanceId,
      String column) {
    return backend.getId() + '|' + tableName + '|' + segmentName + '|' + segmentInstanceId + '|' + column;
  }

  /// A unique id for this loaded segment instance; a reloaded segment is a new instance and gets a new id.
  public synchronized long instanceId(IndexSegment segment) {
    return _segmentInstanceIds.computeIfAbsent(segment, k -> _nextSegmentInstanceId++);
  }

  /// Returns a lease on the resident column for `key`, loading it with `loader` on a miss, or `null` when a column of
  /// `estimatedBytes` cannot be resident within the budget (the caller should use the CPU path instead of loading it
  /// for a single query).
  @Nullable
  public Lease acquire(String key, long estimatedBytes, Supplier<GpuColumn> loader) {
    while (true) {
      CompletableFuture<Entry> pending;
      CompletableFuture<Entry> ownLoad = null;
      synchronized (this) {
        if (estimatedBytes > _maxBytes) {
          return null;
        }
        Entry entry = _entries.get(key);
        if (entry != null) {
          entry._refCount++;
          return new Lease(entry);
        }
        pending = _loading.get(key);
        if (pending == null) {
          ownLoad = new CompletableFuture<>();
          _loading.put(key, ownLoad);
          pending = ownLoad;
        }
      }
      if (ownLoad != null) {
        return load(key, loader, ownLoad);
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
          return new Lease(loaded);
        }
      }
      // Evicted and freed between the load and this waiter; try again.
    }
  }

  private Lease load(String key, Supplier<GpuColumn> loader, CompletableFuture<Entry> ownLoad) {
    Entry entry;
    try {
      entry = new Entry(loader.get());
    } catch (RuntimeException | Error e) {
      synchronized (this) {
        _loading.remove(key);
      }
      ownLoad.completeExceptionally(e);
      throw e;
    }
    List<GpuColumn> toFree;
    synchronized (this) {
      _loading.remove(key);
      _entries.put(key, entry);
      _residentBytes += entry._column.getSizeInBytes();
      entry._refCount++;
      toFree = evictToFit();
    }
    ownLoad.complete(entry);
    freeAll(toFree);
    return new Lease(entry);
  }

  public void setMaxBytes(long maxBytes) {
    List<GpuColumn> toFree;
    synchronized (this) {
      _maxBytes = maxBytes;
      toFree = evictToFit();
    }
    freeAll(toFree);
  }

  public synchronized long getResidentBytes() {
    return _residentBytes;
  }

  public synchronized int getNumResidentColumns() {
    return _entries.size();
  }

  /// Called under the monitor; returns the evicted columns that are unused and must be freed outside it.
  private List<GpuColumn> evictToFit() {
    List<GpuColumn> toFree = new ArrayList<>();
    Iterator<Entry> iterator = _entries.values().iterator();
    while (_residentBytes > _maxBytes && iterator.hasNext()) {
      Entry eldest = iterator.next();
      iterator.remove();
      _residentBytes -= eldest._column.getSizeInBytes();
      eldest._evicted = true;
      if (eldest.markClosedIfUnused()) {
        toFree.add(eldest._column);
      }
    }
    return toFree;
  }

  private static void freeAll(List<GpuColumn> columns) {
    for (GpuColumn column : columns) {
      column.close();
    }
  }

  /// Frees every idle column; columns still leased are freed when released.
  @Override
  public void close() {
    List<GpuColumn> toFree = new ArrayList<>();
    synchronized (this) {
      for (Entry entry : _entries.values()) {
        entry._evicted = true;
        if (entry.markClosedIfUnused()) {
          toFree.add(entry._column);
        }
      }
      _entries.clear();
      _residentBytes = 0;
    }
    freeAll(toFree);
  }
}

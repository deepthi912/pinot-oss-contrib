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

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.pinot.plugin.query.gpu.JavaGpuBackendProvider.JavaColumn;
import org.apache.pinot.segment.spi.IndexSegment;
import org.testng.annotations.Test;

import static org.mockito.Mockito.mock;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;


public class GpuColumnCacheTest {
  private static final long WAIT_NANOS = TimeUnit.SECONDS.toNanos(10);
  private static final long WINDOW_MS = 1_000;
  private final AtomicLong _nowNs = new AtomicLong();

  private final IndexSegment _segment = mock(IndexSegment.class);
  private long _instanceId;

  private GpuColumnCache cache(long maxBytes) {
    GpuColumnCache cache = new GpuColumnCache(maxBytes, WINDOW_MS, _nowNs::get);
    _instanceId = cache.getOrAssignInstanceId(_segment);
    return cache;
  }

  private void idle() {
    _nowNs.addAndGet(TimeUnit.MILLISECONDS.toNanos(WINDOW_MS + 1));
  }

  private static JavaColumn column(int ints) {
    return new JavaColumn(new int[ints]);
  }

  @Test
  public void testHitReusesColumn() {
    GpuColumnCache cache = cache(1000);
    AtomicInteger loads = new AtomicInteger();
    JavaColumn first;
    try (GpuColumnCache.Lease lease = cache.acquire("a", _instanceId, 40, WAIT_NANOS, () -> {
      loads.incrementAndGet();
      return column(10);
    })) {
      first = (JavaColumn) lease.getColumn();
    }
    try (GpuColumnCache.Lease lease = cache.acquire("a", _instanceId, 40, WAIT_NANOS, () -> {
      loads.incrementAndGet();
      return column(10);
    })) {
      assertSame(lease.getColumn(), first);
    }
    assertEquals(loads.get(), 1);
    assertEquals(cache.getResidentBytes(), 40);
  }

  @Test
  public void testLruEvictionFreesIdleColumns() {
    GpuColumnCache cache = cache(100);
    JavaColumn a = column(10);
    JavaColumn b = column(10);
    cache.acquire("a", _instanceId, 40, WAIT_NANOS, () -> a).close();
    cache.acquire("b", _instanceId, 40, WAIT_NANOS, () -> b).close();
    idle();
    cache.acquire("a", _instanceId, 40, WAIT_NANOS, () -> {
      throw new AssertionError("a must still be resident");
    }).close();
    cache.acquire("c", _instanceId, 40, WAIT_NANOS, () -> column(10)).close();
    assertTrue(b.isClosed());
    assertFalse(a.isClosed());
    assertEquals(cache.getNumResidentColumns(), 2);
    assertEquals(cache.getResidentBytes(), 80);
  }

  @Test
  public void testLeasedColumnIsFreedOnlyAfterRelease() {
    GpuColumnCache cache = cache(50);
    JavaColumn a = column(10);
    GpuColumnCache.Lease lease = cache.acquire("a", _instanceId, 40, WAIT_NANOS, () -> a);
    idle();
    // "a" is evicted for "b", but its bytes stay counted while leased, so "b" still does not fit.
    assertNull(cache.acquire("b", _instanceId, 40, WAIT_NANOS, () -> column(10)));
    assertFalse(a.isClosed());
    assertEquals(((JavaColumn) lease.getColumn()).values().length, 10);
    lease.close();
    assertTrue(a.isClosed());
    lease.close();
    cache.acquire("b", _instanceId, 40, WAIT_NANOS, () -> column(10)).close();
  }

  @Test
  public void testColumnLargerThanBudgetIsRefused() {
    GpuColumnCache cache = cache(100);
    assertNull(cache.acquire("big", _instanceId, 101, WAIT_NANOS, () -> {
      throw new AssertionError("must not load a column that cannot be resident");
    }));
    assertEquals(cache.getNumResidentColumns(), 0);
  }

  @Test
  public void testRecentlyUsedColumnsAreNotEvicted() {
    GpuColumnCache cache = cache(100);
    JavaColumn a = column(10);
    JavaColumn b = column(10);
    cache.acquire("a", _instanceId, 40, WAIT_NANOS, () -> a).close();
    cache.acquire("b", _instanceId, 40, WAIT_NANOS, () -> b).close();
    assertNull(cache.acquire("c", _instanceId, 40, WAIT_NANOS, () -> {
      throw new AssertionError("must not load when only recently used columns could be evicted");
    }));
    assertFalse(a.isClosed());
    assertFalse(b.isClosed());
    assertEquals(cache.getNumResidentColumns(), 2);
  }

  @Test
  public void testDenialWithoutRoomEvictsNothing() {
    GpuColumnCache cache = cache(100);
    JavaColumn a = column(10);
    JavaColumn b = column(10);
    cache.acquire("a", _instanceId, 40, WAIT_NANOS, () -> a).close();
    idle();
    cache.acquire("b", _instanceId, 40, WAIT_NANOS, () -> b).close();
    // Needs 80 bytes: idle "a" frees only 40 and recently used "b" is protected, so the load is denied up front,
    // without evicting "a" for nothing.
    assertNull(cache.acquire("c", _instanceId, 80, WAIT_NANOS, () -> {
      throw new AssertionError("must not load");
    }));
    assertFalse(a.isClosed());
    assertFalse(b.isClosed());
  }

  @Test
  public void testEvictedLeasedColumnsCountAgainstBudget() {
    GpuColumnCache cache = cache(100);
    GpuColumnCache.Lease lease = cache.acquire("a", _instanceId, 40, WAIT_NANOS, () -> column(10));
    idle();
    cache.acquire("b", _instanceId, 40, WAIT_NANOS, () -> column(10)).close();
    idle();
    // "a" (still leased) and "b" are evicted for "c"; a's bytes stay counted until its lease is released.
    cache.acquire("c", _instanceId, 60, WAIT_NANOS, () -> column(15)).close();
    idle();
    assertNull(cache.acquire("d", _instanceId, 61, WAIT_NANOS, () -> {
      throw new AssertionError("leased bytes must still count");
    }));
    lease.close();
    cache.acquire("d", _instanceId, 61, WAIT_NANOS, () -> column(15)).close();
  }

  @Test
  public void testReleaseSegmentFreesItsColumns() {
    GpuColumnCache cache = cache(1000);
    JavaColumn a = column(10);
    cache.acquire("a", _instanceId, 40, WAIT_NANOS, () -> a).close();
    cache.releaseSegment(_segment);
    assertTrue(a.isClosed());
    assertEquals(cache.getNumResidentColumns(), 0);
  }

  @Test
  public void testColumnsOfUnloadedSegmentsAreReclaimedFirst() {
    GpuColumnCache cache = cache(100);
    IndexSegment unloaded = mock(IndexSegment.class);
    long unloadedId = cache.getOrAssignInstanceId(unloaded);
    JavaColumn stale = column(10);
    cache.acquire("stale", unloadedId, 60, WAIT_NANOS, () -> stale).close();
    // Recently used, but its segment instance is gone: reclaimed for a live segment's column.
    cache.releaseSegment(unloaded);
    cache.acquire("live", _instanceId, 60, WAIT_NANOS, () -> column(15)).close();
    assertTrue(stale.isClosed());
  }

  @Test
  public void testColumnsOfGarbageCollectedSegmentsAreReclaimed() {
    GpuColumnCache cache = cache(100);
    JavaColumn stale = loadForUnreachableSegment(cache);
    // No releaseSegment: the server path relies on the segment instance being garbage collected.
    long deadlineMs = System.currentTimeMillis() + 30_000;
    GpuColumnCache.Lease lease;
    while ((lease = cache.acquire("live", _instanceId, 60, WAIT_NANOS, () -> column(15))) == null) {
      assertTrue(System.currentTimeMillis() < deadlineMs, "column of the collected segment was never reclaimed");
      System.gc();
    }
    lease.close();
    assertTrue(stale.isClosed());
    assertEquals(cache.getNumResidentColumns(), 1);
  }

  private JavaColumn loadForUnreachableSegment(GpuColumnCache cache) {
    long instanceId = cache.getOrAssignInstanceId(mock(IndexSegment.class));
    JavaColumn stale = column(15);
    cache.acquire("stale", instanceId, 60, WAIT_NANOS, () -> stale).close();
    return stale;
  }

  @Test
  public void testConcurrentLoadsAreCapped()
      throws Exception {
    GpuColumnCache cache = cache(1000);
    CountDownLatch loading = new CountDownLatch(GpuColumnCache.MAX_CONCURRENT_LOADS);
    CountDownLatch release = new CountDownLatch(1);
    ExecutorService executor = Executors.newFixedThreadPool(GpuColumnCache.MAX_CONCURRENT_LOADS);
    try {
      for (int i = 0; i < GpuColumnCache.MAX_CONCURRENT_LOADS; i++) {
        String key = "slow" + i;
        executor.submit(() -> cache.acquire(key, _instanceId, 40, WAIT_NANOS, () -> {
          loading.countDown();
          try {
            release.await();
          } catch (InterruptedException e) {
            throw new IllegalStateException(e);
          }
          return column(10);
        }).close());
      }
      assertTrue(loading.await(10, TimeUnit.SECONDS));
      AtomicBoolean extraLoaded = new AtomicBoolean();
      CompletableFuture<GpuColumnCache.Lease> extra = CompletableFuture.supplyAsync(
          () -> cache.acquire("extra", _instanceId, 40, WAIT_NANOS, () -> {
            extraLoaded.set(true);
            return column(10);
          }));
      // The third load waits for a slot instead of skipping the GPU.
      Thread.sleep(200);
      assertFalse(extraLoaded.get(), "a third load started while all load slots were busy");
      release.countDown();
      GpuColumnCache.Lease lease = extra.get(10, TimeUnit.SECONDS);
      assertNotNull(lease);
      assertTrue(extraLoaded.get());
      lease.close();
    } finally {
      release.countDown();
      executor.shutdown();
      assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
    }
  }

  @Test
  public void testLoadSlotWaitIsBounded()
      throws Exception {
    GpuColumnCache cache = cache(1000);
    CountDownLatch loading = new CountDownLatch(GpuColumnCache.MAX_CONCURRENT_LOADS);
    CountDownLatch release = new CountDownLatch(1);
    ExecutorService executor = Executors.newFixedThreadPool(GpuColumnCache.MAX_CONCURRENT_LOADS);
    try {
      for (int i = 0; i < GpuColumnCache.MAX_CONCURRENT_LOADS; i++) {
        String key = "slow" + i;
        executor.submit(() -> cache.acquire(key, _instanceId, 40, WAIT_NANOS, () -> {
          loading.countDown();
          awaitQuietly(release);
          return column(10);
        }).close());
      }
      assertTrue(loading.await(10, TimeUnit.SECONDS));
      // A query whose deadline passes while waiting for a load slot runs on the CPU.
      assertNull(cache.acquire("extra", _instanceId, 40, TimeUnit.MILLISECONDS.toNanos(50), () -> {
        throw new AssertionError("must not load after the wait expired");
      }));
    } finally {
      release.countDown();
      executor.shutdown();
      assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
    }
  }

  @Test
  public void testWaiterLoadsItselfWhenAnotherLoadOfTheKeyFails()
      throws Exception {
    GpuColumnCache cache = cache(1000);
    CountDownLatch loading = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);
    // E.g. the first query was cancelled mid-upload: the waiting query must not inherit that failure.
    CompletableFuture<Void> failing = CompletableFuture.runAsync(() -> assertThrows(IllegalStateException.class,
        () -> cache.acquire("k", _instanceId, 40, WAIT_NANOS, () -> {
          loading.countDown();
          awaitQuietly(release);
          throw new IllegalStateException("query cancelled");
        })));
    assertTrue(loading.await(10, TimeUnit.SECONDS));
    JavaColumn own = column(10);
    CompletableFuture<GpuColumnCache.Lease> waiter =
        CompletableFuture.supplyAsync(() -> cache.acquire("k", _instanceId, 40, WAIT_NANOS, () -> own));
    Thread.sleep(100);
    release.countDown();
    failing.get(10, TimeUnit.SECONDS);
    try (GpuColumnCache.Lease lease = waiter.get(10, TimeUnit.SECONDS)) {
      assertSame(lease.getColumn(), own);
    }
  }

  private static void awaitQuietly(CountDownLatch latch) {
    try {
      latch.await();
    } catch (InterruptedException e) {
      throw new IllegalStateException(e);
    }
  }

  @Test
  public void testLoadFinishingAfterReleaseIsNotCached()
      throws Exception {
    for (boolean closeCache : new boolean[]{false, true}) {
      GpuColumnCache cache = cache(1000);
      CountDownLatch loading = new CountDownLatch(1);
      CountDownLatch release = new CountDownLatch(1);
      JavaColumn column = column(10);
      CompletableFuture<GpuColumnCache.Lease> load =
          CompletableFuture.supplyAsync(() -> cache.acquire("k", _instanceId, 40, WAIT_NANOS, () -> {
            loading.countDown();
            awaitQuietly(release);
            return column;
          }));
      assertTrue(loading.await(10, TimeUnit.SECONDS));
      if (closeCache) {
        cache.close();
      } else {
        cache.releaseSegment(_segment);
      }
      release.countDown();
      GpuColumnCache.Lease lease = load.get(10, TimeUnit.SECONDS);
      assertSame(lease.getColumn(), column);
      assertEquals(cache.getNumResidentColumns(), 0);
      lease.close();
      assertTrue(column.isClosed(), "the column of a released segment must be freed with its last lease");
      if (closeCache) {
        assertNull(cache.acquire("k2", _instanceId, 40, WAIT_NANOS, () -> {
          throw new AssertionError("a closed cache must not load");
        }));
      }
    }
  }

  @Test
  public void testFailedLoadPropagatesAndNextAcquireRetries() {
    GpuColumnCache cache = cache(100);
    assertThrows(IllegalStateException.class, () -> cache.acquire("a", _instanceId, 40, WAIT_NANOS, () -> {
      throw new IllegalStateException("device out of memory");
    }));
    try (GpuColumnCache.Lease lease = cache.acquire("a", _instanceId, 40, WAIT_NANOS, () -> column(10))) {
      assertEquals(lease.getColumn().getNumValues(), 10);
    }
  }

  @Test
  public void testConcurrentMissesShareOneLoad()
      throws Exception {
    GpuColumnCache cache = cache(1000);
    AtomicInteger loads = new AtomicInteger();
    CountDownLatch loading = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);
    ExecutorService executor = Executors.newFixedThreadPool(4);
    try {
      Future<?>[] futures = new Future<?>[4];
      for (int i = 0; i < futures.length; i++) {
        futures[i] = executor.submit(() -> {
          try (GpuColumnCache.Lease lease = cache.acquire("a", _instanceId, 40, WAIT_NANOS, () -> {
            loads.incrementAndGet();
            loading.countDown();
            try {
              release.await();
            } catch (InterruptedException e) {
              throw new IllegalStateException(e);
            }
            return column(10);
          })) {
            assertEquals(lease.getColumn().getNumValues(), 10);
          }
          return null;
        });
      }
      assertTrue(loading.await(10, TimeUnit.SECONDS));
      cache.acquire("other", _instanceId, 40, WAIT_NANOS, () -> column(10)).close();
      release.countDown();
      for (Future<?> future : futures) {
        future.get(10, TimeUnit.SECONDS);
      }
    } finally {
      executor.shutdownNow();
    }
    assertEquals(loads.get(), 1);
  }

  @Test
  public void testCloseFreesIdleNowAndLeasedOnRelease() {
    GpuColumnCache cache = cache(1000);
    JavaColumn idle = column(10);
    JavaColumn leased = column(10);
    cache.acquire("idle", _instanceId, 40, WAIT_NANOS, () -> idle).close();
    GpuColumnCache.Lease lease = cache.acquire("leased", _instanceId, 40, WAIT_NANOS, () -> leased);
    cache.close();
    assertTrue(idle.isClosed());
    assertFalse(leased.isClosed());
    lease.close();
    assertTrue(leased.isClosed());
  }
}

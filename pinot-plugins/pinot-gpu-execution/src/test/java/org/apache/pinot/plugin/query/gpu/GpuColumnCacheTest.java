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

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.pinot.plugin.query.gpu.JavaGpuBackendProvider.JavaColumn;
import org.testng.annotations.Test;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;


public class GpuColumnCacheTest {
  private static final long WINDOW_MS = 1_000;
  private final AtomicLong _nowNs = new AtomicLong();

  private GpuColumnCache cache(long maxBytes) {
    return new GpuColumnCache(maxBytes, WINDOW_MS, _nowNs::get);
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
    try (GpuColumnCache.Lease lease = cache.acquire("a", 40, () -> {
      loads.incrementAndGet();
      return column(10);
    })) {
      first = (JavaColumn) lease.getColumn();
    }
    try (GpuColumnCache.Lease lease = cache.acquire("a", 40, () -> {
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
    cache.acquire("a", 40, () -> a).close();
    cache.acquire("b", 40, () -> b).close();
    idle();
    cache.acquire("a", 40, () -> {
      throw new AssertionError("a must still be resident");
    }).close();
    cache.acquire("c", 40, () -> column(10)).close();
    assertTrue(b.isClosed());
    assertFalse(a.isClosed());
    assertEquals(cache.getNumResidentColumns(), 2);
    assertEquals(cache.getResidentBytes(), 80);
  }

  @Test
  public void testLeasedColumnIsFreedOnlyAfterRelease() {
    GpuColumnCache cache = cache(50);
    JavaColumn a = column(10);
    GpuColumnCache.Lease lease = cache.acquire("a", 40, () -> a);
    idle();
    // "a" is evicted for "b", but its bytes stay counted while leased, so "b" still does not fit.
    assertNull(cache.acquire("b", 40, () -> column(10)));
    assertFalse(a.isClosed());
    assertEquals(((JavaColumn) lease.getColumn()).values().length, 10);
    lease.close();
    assertTrue(a.isClosed());
    lease.close();
    cache.acquire("b", 40, () -> column(10)).close();
  }

  @Test
  public void testColumnLargerThanBudgetIsRefused() {
    GpuColumnCache cache = cache(100);
    assertNull(cache.acquire("big", 101, () -> {
      throw new AssertionError("must not load a column that cannot be resident");
    }));
    assertEquals(cache.getNumResidentColumns(), 0);
  }

  @Test
  public void testRecentlyUsedColumnsAreNotEvicted() {
    GpuColumnCache cache = cache(100);
    JavaColumn a = column(10);
    JavaColumn b = column(10);
    cache.acquire("a", 40, () -> a).close();
    cache.acquire("b", 40, () -> b).close();
    assertNull(cache.acquire("c", 40, () -> {
      throw new AssertionError("must not load when only recently used columns could be evicted");
    }));
    assertFalse(a.isClosed());
    assertFalse(b.isClosed());
    assertEquals(cache.getNumResidentColumns(), 2);
  }

  @Test
  public void testPartialEvictionThenDenialFreesEvicted() {
    GpuColumnCache cache = cache(100);
    JavaColumn a = column(10);
    JavaColumn b = column(10);
    cache.acquire("a", 40, () -> a).close();
    idle();
    cache.acquire("b", 40, () -> b).close();
    // Needs 80 bytes: evicting idle "a" frees 40, then recently used "b" is protected, so the load is denied.
    assertNull(cache.acquire("c", 80, () -> {
      throw new AssertionError("must not load");
    }));
    assertTrue(a.isClosed());
    assertFalse(b.isClosed());
  }

  @Test
  public void testEvictedLeasedColumnsCountAgainstBudget() {
    GpuColumnCache cache = cache(100);
    GpuColumnCache.Lease lease = cache.acquire("a", 40, () -> column(10));
    idle();
    cache.acquire("b", 40, () -> column(10)).close();
    idle();
    // "a" (still leased) and "b" are evicted for "c"; a's bytes stay counted until its lease is released.
    cache.acquire("c", 60, () -> column(15)).close();
    idle();
    assertNull(cache.acquire("d", 61, () -> {
      throw new AssertionError("leased bytes must still count");
    }));
    lease.close();
    cache.acquire("d", 61, () -> column(15)).close();
  }

  @Test
  public void testFailedLoadPropagatesAndNextAcquireRetries() {
    GpuColumnCache cache = cache(100);
    assertThrows(IllegalStateException.class, () -> cache.acquire("a", 40, () -> {
      throw new IllegalStateException("device out of memory");
    }));
    try (GpuColumnCache.Lease lease = cache.acquire("a", 40, () -> column(10))) {
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
          try (GpuColumnCache.Lease lease = cache.acquire("a", 40, () -> {
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
      cache.acquire("other", 40, () -> column(10)).close();
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
    cache.acquire("idle", 40, () -> idle).close();
    GpuColumnCache.Lease lease = cache.acquire("leased", 40, () -> leased);
    cache.close();
    assertTrue(idle.isClosed());
    assertFalse(leased.isClosed());
    lease.close();
    assertTrue(leased.isClosed());
  }
}

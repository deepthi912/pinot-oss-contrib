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

import java.math.BigInteger;
import org.testng.annotations.Test;

import static org.testng.Assert.assertEquals;


/// Host-side parts of the Metal backend that need no GPU; the kernels are covered by
/// `GpuGroupByQueriesTest#testMetalBackendEndToEnd`.
public class MetalGpuBackendTest {
  private static final BigInteger BIAS = BigInteger.ONE.shiftLeft(31);
  private static final BigInteger TWO_64 = BigInteger.ONE.shiftLeft(64);

  @Test
  public void testRemoveSumBias() {
    int[][] groups = {
        {}, {0}, {-1}, {Integer.MIN_VALUE}, {Integer.MAX_VALUE}, {Integer.MIN_VALUE, Integer.MAX_VALUE},
        {-5, -7, 3, Integer.MIN_VALUE, Integer.MIN_VALUE}, {Integer.MAX_VALUE, Integer.MAX_VALUE, 1}
    };
    long[] biasedSums = new long[groups.length];
    long[] counts = new long[groups.length];
    long[] expected = new long[groups.length];
    for (int g = 0; g < groups.length; g++) {
      BigInteger biased = BigInteger.ZERO;
      for (int value : groups[g]) {
        biased = biased.add(BigInteger.valueOf(value).add(BIAS));
        expected[g] += value;
      }
      biasedSums[g] = biased.mod(TWO_64).longValue();
      counts[g] = groups[g].length;
    }
    MetalGpuBackend.removeSumBias(biasedSums, counts);
    assertEquals(biasedSums, expected);
  }

  @Test
  public void testRemoveSumBiasWithLargeCounts() {
    // 3 billion rows: the biased sum of all-MAX values exceeds 2^63 and wraps, the true sums still fit in a long.
    long count = 3_000_000_000L;
    long[] counts = {count, count, count};
    long[] biasedSums = {
        biased(count, Integer.MAX_VALUE), biased(count, Integer.MIN_VALUE), biased(count, -1)
    };
    MetalGpuBackend.removeSumBias(biasedSums, counts);
    assertEquals(biasedSums, new long[]{count * Integer.MAX_VALUE, count * Integer.MIN_VALUE, -count});
  }

  /// What the 64-bit unsigned device accumulator holds after `count` rows of `value`.
  private static long biased(long count, int value) {
    return BigInteger.valueOf(count).multiply(BigInteger.valueOf(value).add(BIAS)).mod(TWO_64).longValue();
  }
}

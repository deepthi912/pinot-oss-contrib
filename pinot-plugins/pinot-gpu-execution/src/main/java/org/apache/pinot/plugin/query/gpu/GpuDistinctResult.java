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

import org.roaringbitmap.RoaringBitmap;

/// Distinct (group, value id) pairs from [GpuBackend#distinct] as a bitmap: bit `group * cardinality + id` is set when
/// the pair occurs among matching documents. Immutable.
public final class GpuDistinctResult {
  private final int[] _words;
  private final long _deviceTimeNs;

  public GpuDistinctResult(int[] words, long deviceTimeNs) {
    _words = words;
    _deviceTimeNs = deviceTimeNs;
  }

  /// The ids marked for `group`, visiting only set bits.
  public RoaringBitmap getIds(int group, int cardinality) {
    RoaringBitmap ids = new RoaringBitmap();
    long from = (long) group * cardinality;
    long to = from + cardinality;
    for (int w = (int) (from >>> 5); w <= (int) ((to - 1) >>> 5); w++) {
      int word = _words[w];
      while (word != 0) {
        long bit = ((long) w << 5) + Integer.numberOfTrailingZeros(word);
        if (bit >= from && bit < to) {
          ids.add((int) (bit - from));
        }
        word &= word - 1;
      }
    }
    return ids;
  }

  /// Whether each group has at least one id marked.
  public boolean[] nonEmptyGroups(int numGroups, int cardinality) {
    boolean[] nonEmpty = new boolean[numGroups];
    for (int w = 0; w < _words.length; w++) {
      int word = _words[w];
      while (word != 0) {
        long bit = ((long) w << 5) + Integer.numberOfTrailingZeros(word);
        int group = (int) (bit / cardinality);
        nonEmpty[group] = true;
        // Skip the rest of this group's bits within the word.
        long nextGroupBit = (long) (group + 1) * cardinality;
        long wordEnd = ((long) w << 5) + 32;
        word = nextGroupBit >= wordEnd ? 0 : word & (-1 << (int) (nextGroupBit - ((long) w << 5)));
      }
    }
    return nonEmpty;
  }

  public long getDeviceTimeNs() {
    return _deviceTimeNs;
  }
}

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

import java.nio.ByteBuffer;
import java.util.Random;
import org.roaringbitmap.buffer.ImmutableRoaringBitmap;
import org.roaringbitmap.buffer.MappeableContainerPointer;
import org.roaringbitmap.buffer.MutableRoaringBitmap;
import org.testng.annotations.Test;

import static org.testng.Assert.assertEquals;


public class GpuSegmentAggregatorTest {

  @Test
  public void testFilterWordsMatchPerDocumentConversion() {
    Random random = new Random(42);
    boolean[] seenKinds = new boolean[3];
    for (int numDocs : new int[]{1, 31, 33, 65_535, 65_536, 65_537, 200_003, 262_144, 300_017}) {
      for (int round = 0; round < 5; round++) {
        MutableRoaringBitmap bitmap = randomBitmap(random, numDocs);
        recordContainerKinds(bitmap, seenKinds);
        int[] expected = naiveWords(bitmap, numDocs);
        assertEquals(GpuSegmentAggregator.toWords(bitmap, numDocs), expected, "numDocs " + numDocs);
        // Memory-mapped containers (as read from an index) take the buffer-backed code paths.
        assertEquals(GpuSegmentAggregator.toWords(mapped(bitmap), numDocs), expected, "mapped, numDocs " + numDocs);
      }
    }
    assertEquals(seenKinds, new boolean[]{true, true, true}, "array, bitmap and run containers must all be covered");
  }

  private static void recordContainerKinds(ImmutableRoaringBitmap bitmap, boolean[] seenKinds) {
    for (MappeableContainerPointer pointer = bitmap.getContainerPointer(); pointer.hasContainer();
        pointer.advance()) {
      seenKinds[pointer.isBitmapContainer() ? 1 : pointer.isRunContainer() ? 2 : 0] = true;
    }
  }

  /// Per 64K chunk: sparse (array container), dense (bitmap container) or long ranges (run containers).
  private static MutableRoaringBitmap randomBitmap(Random random, int numDocs) {
    MutableRoaringBitmap bitmap = new MutableRoaringBitmap();
    for (int chunkStart = 0; chunkStart < numDocs; chunkStart += 1 << 16) {
      int chunkEnd = Math.min(numDocs, chunkStart + (1 << 16));
      switch (random.nextInt(4)) {
        case 0:
          for (int i = 0; i < 100; i++) {
            bitmap.add(chunkStart + random.nextInt(chunkEnd - chunkStart));
          }
          break;
        case 1:
          for (int docId = chunkStart; docId < chunkEnd; docId++) {
            if (random.nextInt(3) != 0) {
              bitmap.add(docId);
            }
          }
          break;
        case 2:
          for (int i = 0; i < 20; i++) {
            long from = chunkStart + random.nextInt(chunkEnd - chunkStart);
            bitmap.add(from, Math.min(chunkEnd, from + 1 + random.nextInt(3000)));
          }
          // Runs ending exactly at the segment end and covering a whole chunk.
          bitmap.add((long) chunkEnd - 1, chunkEnd);
          break;
        default:
          bitmap.add((long) chunkStart, chunkEnd);
          break;
      }
    }
    bitmap.runOptimize();
    return bitmap;
  }

  private static ImmutableRoaringBitmap mapped(MutableRoaringBitmap bitmap) {
    ByteBuffer buffer = ByteBuffer.allocateDirect(bitmap.serializedSizeInBytes());
    bitmap.serialize(buffer);
    buffer.flip();
    return new ImmutableRoaringBitmap(buffer);
  }

  private static int[] naiveWords(ImmutableRoaringBitmap docIds, int numDocs) {
    int[] words = new int[(numDocs + 31) >>> 5];
    docIds.forEach((int docId) -> words[docId >>> 5] |= 1 << docId);
    return words;
  }
}

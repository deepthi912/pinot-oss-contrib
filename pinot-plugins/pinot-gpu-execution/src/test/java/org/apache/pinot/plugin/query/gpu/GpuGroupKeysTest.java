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

import org.testng.annotations.Test;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;


public class GpuGroupKeysTest {

  @Test
  public void testEncodeAndDecodeFirstColumnMostSignificant() {
    int[] cardinalities = {3, 4, 5};
    GpuGroupKeys keys = new GpuGroupKeys(cardinalities);
    assertEquals(keys.getNumGroups(), 60);

    // Every combination, enumerated in lexicographic order, must get consecutive ids.
    int[][] dictIds = new int[3][60];
    int row = 0;
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 4; b++) {
        for (int c = 0; c < 5; c++) {
          dictIds[0][row] = a;
          dictIds[1][row] = b;
          dictIds[2][row] = c;
          row++;
        }
      }
    }
    int[] ids = new int[60];
    keys.encode(dictIds, 60, ids);
    for (int i = 0; i < 60; i++) {
      assertEquals(ids[i], i);
      for (int column = 0; column < 3; column++) {
        assertEquals(keys.decodeDictId(ids[i], column), dictIds[column][i]);
      }
    }
    // Only the first `length` rows are written.
    int[] partial = new int[60];
    keys.encode(dictIds, 7, partial);
    assertEquals(partial[6], 6);
    assertEquals(partial[7], 0);
  }

  @Test
  public void testSingleColumnKeyIsTheDictId() {
    GpuGroupKeys keys = new GpuGroupKeys(new int[]{9});
    int[] ids = new int[3];
    keys.encode(new int[][]{{8, 0, 4}}, 3, ids);
    assertEquals(ids, new int[]{8, 0, 4});
    assertEquals(keys.decodeDictId(8, 0), 8);
  }

  @Test
  public void testCardinalityProductAndBound() {
    assertEquals(GpuGroupKeys.cardinalityProduct(new int[]{1000, 100}), 100_000L);
    assertEquals(GpuGroupKeys.cardinalityProduct(new int[]{0, 7}), 0L);
    assertTrue(GpuGroupKeys.cardinalityProduct(new int[]{Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE})
        > Integer.MAX_VALUE);
    assertEquals(new GpuGroupKeys(new int[]{1 << 10, 1 << 10}).getNumGroups(), GpuGroupKeys.MAX_DENSE_GROUPS);
    assertThrows(IllegalArgumentException.class, () -> new GpuGroupKeys(new int[]{17, 61_681}));
    assertThrows(IllegalArgumentException.class, () -> new GpuGroupKeys(new int[]{1 << 16, 1 << 16}));
  }
}

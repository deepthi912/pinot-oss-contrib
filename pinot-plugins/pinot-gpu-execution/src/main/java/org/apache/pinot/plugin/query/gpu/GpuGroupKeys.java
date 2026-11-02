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

import com.google.common.base.Preconditions;
import org.apache.pinot.segment.spi.IndexSegment;


/// Dense group ids over the dictionary ids of the GROUP BY columns: `id = sum(dictId_i * stride_i)` with the first
/// column most significant and the last one of stride 1. With sorted dictionaries, ascending ids follow the
/// lexicographic order of the group values. Immutable.
final class GpuGroupKeys {
  /// Maximum number of groups of the dense combined GROUP BY key: bounds each segment's per-group counts and sums
  /// (one `long` per group) to 8 MB per array.
  static final int MAX_DENSE_GROUPS = 1 << 20;

  private final int[] _cardinalities;
  private final int[] _strides;
  private final int _numGroups;

  GpuGroupKeys(int[] cardinalities) {
    Preconditions.checkArgument(cardinalities.length > 0, "No group-by column");
    long numGroups = cardinalityProduct(cardinalities);
    Preconditions.checkArgument(numGroups <= MAX_DENSE_GROUPS, "Too many groups: %s", numGroups);
    _cardinalities = cardinalities.clone();
    _strides = new int[cardinalities.length];
    int stride = 1;
    for (int i = cardinalities.length - 1; i >= 0; i--) {
      _strides[i] = stride;
      stride *= cardinalities[i];
    }
    _numGroups = (int) numGroups;
  }

  /// Dictionary sizes of `columns` in `segment`; every column must have a dictionary.
  static int[] cardinalities(IndexSegment segment, String[] columns) {
    int[] cardinalities = new int[columns.length];
    for (int i = 0; i < columns.length; i++) {
      cardinalities[i] = segment.getDataSource(columns[i]).getDictionary().length();
    }
    return cardinalities;
  }

  /// Product of the cardinalities; any value above [Integer#MAX_VALUE] means "too many" and is not exact.
  static long cardinalityProduct(int[] cardinalities) {
    long product = 1;
    for (int cardinality : cardinalities) {
      product *= cardinality;
      if (product > Integer.MAX_VALUE) {
        return product;
      }
    }
    return product;
  }

  int getNumGroups() {
    return _numGroups;
  }

  /// Writes the group ids of `length` rows, given each column's dictionary ids, to `groupIds`.
  void encode(int[][] dictIds, int length, int[] groupIds) {
    System.arraycopy(dictIds[0], 0, groupIds, 0, length);
    for (int c = 1; c < dictIds.length; c++) {
      int cardinality = _cardinalities[c];
      int[] columnDictIds = dictIds[c];
      for (int i = 0; i < length; i++) {
        groupIds[i] = groupIds[i] * cardinality + columnDictIds[i];
      }
    }
  }

  int decodeDictId(int groupId, int column) {
    return (groupId / _strides[column]) % _cardinalities[column];
  }
}

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
package org.apache.pinot.core.operator.filter.predicate;

import java.util.regex.Matcher;
import org.apache.pinot.common.request.context.predicate.Predicate;
import org.apache.pinot.common.request.context.predicate.RegexpLikePredicate;
import org.apache.pinot.segment.spi.index.reader.Dictionary;


/// Base class for dictionary-based REGEXP_LIKE predicate evaluators that use dictionary IDs for matching.
public abstract class BaseDictIdBasedRegexpLikePredicateEvaluator extends BaseDictionaryBasedPredicateEvaluator {

  // Reused matcher for evaluating the regex against raw string values from the forward index. Used only on the
  // scan path when the forward index is RAW (no dict-encoded forward stream is available) and no inverted/sorted
  // index can serve the predicate, so the IFST/FST-precomputed matching-dict-id set cannot be leveraged. Not
  // thread-safe; the evaluator is expected to be used by a single thread (one query, one segment slice).
  private final Matcher _rawValueMatcher;

  protected BaseDictIdBasedRegexpLikePredicateEvaluator(Predicate predicate, Dictionary dictionary) {
    super(predicate, dictionary);
    _rawValueMatcher = ((RegexpLikePredicate) predicate).getPattern().matcher("");
  }

  /// Evaluate the predicate against a raw string value (forward-index value), bypassing the dictionary. This is the
  /// fallback path used by the scan filter operator when the forward index is RAW but a dictionary still exists for
  /// secondary indexes (e.g. IFST); the matcher in {@link
  /// org.apache.pinot.core.operator.dociditerators.SVScanDocIdIterator} reads raw strings and calls this method
  /// instead of {@code applySV(int dictId)}.
  @Override
  public boolean applySV(String value) {
    if (_alwaysFalse) {
      return false;
    }
    if (_alwaysTrue) {
      return true;
    }
    return _rawValueMatcher.reset(value).find();
  }

  @Override
  public boolean applyMV(String[] values, int length) {
    if (_alwaysFalse) {
      return false;
    }
    if (_alwaysTrue) {
      return true;
    }
    for (int i = 0; i < length; i++) {
      if (_rawValueMatcher.reset(values[i]).find()) {
        return true;
      }
    }
    return false;
  }
}

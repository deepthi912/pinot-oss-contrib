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

import org.apache.pinot.common.request.context.ExpressionContext;
import org.apache.pinot.common.request.context.predicate.RegexpLikePredicate;
import org.apache.pinot.segment.spi.index.reader.Dictionary;
import org.mockito.Mockito;
import org.testng.annotations.Test;

import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;


/**
 * Verifies that dict-id based REGEXP_LIKE evaluators (IFST/FST/DictId) correctly evaluate raw string values
 * via the regex matcher when the scan iterator calls {@code applySV(String)}. This path is taken when the
 * forward index is RAW and a separate dictionary exists, but no inverted/sorted index is available — i.e.
 * the planner falls through to {@code ScanBasedFilterOperator} carrying a dict-based evaluator.
 *
 * Prior to this fix, the inherited {@code applySV(String)} from {@code BaseDictionaryBasedPredicateEvaluator}
 * threw {@code UnsupportedOperationException} and the query crashed.
 */
public class BaseDictIdBasedRegexpLikePredicateEvaluatorTest {

  @Test
  public void applySVStringMatchesCaseSensitiveRegex() {
    TestEvaluator evaluator = newEvaluator("^foo");

    assertTrue(evaluator.applySV("foobar"));
    assertTrue(evaluator.applySV("foo"));
    assertFalse(evaluator.applySV("FOO"));        // case-sensitive
    assertFalse(evaluator.applySV("barfoo"));     // not at start
    assertFalse(evaluator.applySV(""));
  }

  @Test
  public void applySVStringMatchesCaseInsensitiveRegex() {
    TestEvaluator evaluator = newEvaluator("^foo", "i");

    assertTrue(evaluator.applySV("foobar"));
    assertTrue(evaluator.applySV("FOObar"));
    assertTrue(evaluator.applySV("FoOzilla"));
    assertFalse(evaluator.applySV("barfoo"));
  }

  @Test
  public void applyMVStringReturnsTrueIfAnyMatches() {
    TestEvaluator evaluator = newEvaluator("^foo");

    assertTrue(evaluator.applyMV(new String[]{"bar", "baz", "foobar"}, 3));
    assertFalse(evaluator.applyMV(new String[]{"bar", "baz", "qux"}, 3));
    // Length param is respected — only the first 2 are inspected.
    assertFalse(evaluator.applyMV(new String[]{"bar", "baz", "foobar"}, 2));
  }

  @Test
  public void alwaysFalseShortCircuitsRawValuePath() {
    TestEvaluator evaluator = newEvaluator("^foo");
    evaluator.setAlwaysFalse();

    assertFalse(evaluator.applySV("foobar"));
    assertFalse(evaluator.applyMV(new String[]{"foobar"}, 1));
  }

  @Test
  public void alwaysTrueShortCircuitsRawValuePath() {
    TestEvaluator evaluator = newEvaluator("^foo");
    evaluator.setAlwaysTrue();

    assertTrue(evaluator.applySV("anything"));
    assertTrue(evaluator.applyMV(new String[]{"anything"}, 1));
  }

  private static TestEvaluator newEvaluator(String regex) {
    return newEvaluator(regex, null);
  }

  private static TestEvaluator newEvaluator(String regex, String matchParameter) {
    RegexpLikePredicate predicate = matchParameter == null
        ? new RegexpLikePredicate(ExpressionContext.forIdentifier("col"), regex)
        : new RegexpLikePredicate(ExpressionContext.forIdentifier("col"), regex, matchParameter);
    Dictionary dictionary = Mockito.mock(Dictionary.class);
    return new TestEvaluator(predicate, dictionary);
  }

  /// Minimal concrete subclass for the test. We only exercise the raw-value path, so the dict-id path can be a
  /// trivial stub.
  private static class TestEvaluator extends BaseDictIdBasedRegexpLikePredicateEvaluator {

    TestEvaluator(RegexpLikePredicate predicate, Dictionary dictionary) {
      super(predicate, dictionary);
    }

    void setAlwaysTrue() {
      _alwaysTrue = true;
    }

    void setAlwaysFalse() {
      _alwaysFalse = true;
    }

    @Override
    public boolean applySV(int dictId) {
      // Unused on the raw-value path; tests only call applySV(String) / applyMV(String[]).
      return false;
    }
  }
}

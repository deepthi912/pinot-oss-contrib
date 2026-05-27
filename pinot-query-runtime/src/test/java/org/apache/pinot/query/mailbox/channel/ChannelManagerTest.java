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
package org.apache.pinot.query.mailbox.channel;

import java.time.Duration;
import org.apache.pinot.spi.utils.CommonConstants;
import org.testng.annotations.Test;


public class ChannelManagerTest {

  /// Pins the fail-fast gate: a non-positive `writeBufferLowWaterMarkBytes` must throw at startup rather
  /// than surfacing later as a Netty `WriteBufferWaterMark` constructor failure on the first send.
  @Test(expectedExceptions = IllegalArgumentException.class,
      expectedExceptionsMessageRegExp = ".*writeBufferLowWaterMarkBytes must be positive.*")
  public void testConstructorRejectsZeroWriteBufferLowWaterMark() {
    new ChannelManager(null, 4_000_000, Duration.ofDays(365),
        CommonConstants.MultiStageQueryRunner.DEFAULT_GRPC_WRITE_BUFFER_HIGH_WATER_MARK_BYTES,
        0);
  }

  /// Pins the eager `new WriteBufferWaterMark(low, high)` invariant: when `low > high`, Netty's own
  /// constructor throws `IllegalArgumentException`. Constructing the watermark eagerly in
  /// `ChannelManager` is what makes this surface at startup instead of on the first send to a
  /// previously-unseen peer.
  @Test(expectedExceptions = IllegalArgumentException.class)
  public void testConstructorRejectsLowWatermarkAboveHighWatermark() {
    new ChannelManager(null, 4_000_000, Duration.ofDays(365),
        32 * 1024 * 1024,  // high
        64 * 1024 * 1024); // low > high
  }
}

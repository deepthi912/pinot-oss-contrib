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
package org.apache.pinot.core.transport;

import io.netty.channel.ChannelHandler;
import io.netty.channel.socket.SocketChannel;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.pinot.common.config.NettyConfig;
import org.apache.pinot.common.config.TlsConfig;
import org.apache.pinot.common.metrics.ServerMetrics;
import org.apache.pinot.spi.env.PinotConfiguration;
import org.apache.pinot.spi.metrics.PinotMetricUtils;
import org.apache.pinot.spi.metrics.PinotMetricsRegistry;
import org.apache.pinot.util.TestUtils;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import static org.mockito.Mockito.mock;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;


public class QueryServerTest {
  @DataProvider
  public Object[][] parameters() {
    return new Object[][]{
        new Object[]{true}, new Object[]{false},
    };
  }

  @Test(dataProvider = "parameters")
  public void startAndStop(final boolean nativeTransportEnabled) {
    PinotMetricUtils.init(new PinotConfiguration());
    PinotMetricsRegistry registry = PinotMetricUtils.getPinotMetricsRegistry();
    ServerMetrics.register(new ServerMetrics(registry));
    NettyConfig nettyConfig = new NettyConfig();
    nettyConfig.setNativeTransportsEnabled(nativeTransportEnabled);
    TlsConfig tlsConfig = new TlsConfig();
    ChannelHandler channelHandler = mock(ChannelHandler.class);

    QueryServer server = new QueryServer(0, nettyConfig, tlsConfig, channelHandler);
    server.start();

    // The server should use the shared process-wide bounded allocator
    assertSame(server.getChannel().config().getAllocator(),
        PooledByteBufAllocatorWithLimits.getSharedBufferAllocatorWithLimits());

    final InetSocketAddress serverAddress = server.getChannel().localAddress();

    assertTrue(connectionOk(serverAddress));

    server.shutDown();
    assertFalse(connectionOk(serverAddress));
  }

  @Test
  public void testAllChannelsCleanupOnClose()
      throws Exception {
    PinotMetricUtils.init(new PinotConfiguration());
    PinotMetricsRegistry registry = PinotMetricUtils.getPinotMetricsRegistry();
    ServerMetrics.register(new ServerMetrics(registry));
    QueryServer server = new QueryServer(0, new NettyConfig(), null, mock(ChannelHandler.class));
    server.start();

    InetSocketAddress serverAddress = server.getChannel().localAddress();
    Socket socket = new Socket(serverAddress.getHostName(), serverAddress.getPort());

    try {
      TestUtils.waitForCondition(aVoid -> server.getConnectedChannelCount() > 0, 5_000L,
          "Channel was not registered in _allChannels");

      // The accepted child channels (which allocate the request/response buffers) must also use the shared allocator
      SocketChannel connectedChannel = server.getConnectedChannels().iterator().next();
      assertSame(connectedChannel.config().getAllocator(),
          PooledByteBufAllocatorWithLimits.getSharedBufferAllocatorWithLimits());

      socket.close();

      TestUtils.waitForCondition(aVoid -> server.getConnectedChannelCount() == 0, 5_000L,
          "Channel was not removed from _allChannels after close");
    } finally {
      IOUtils.closeQuietly(socket);
      server.shutDown();
    }
  }

  /// A start that fails, e.g. because the port is still held by a stopped server, must report the bind failure and
  /// leave the server able to start once the port is free.
  @Test(dataProvider = "parameters")
  public void testStartAfterFailedStart(boolean nativeTransportEnabled)
      throws Exception {
    PinotMetricUtils.init(new PinotConfiguration());
    ServerMetrics.register(new ServerMetrics(PinotMetricUtils.getPinotMetricsRegistry()));
    NettyConfig nettyConfig = new NettyConfig();
    nettyConfig.setNativeTransportsEnabled(nativeTransportEnabled);

    QueryServer server;
    try (ServerSocket portHolder = new ServerSocket(0)) {
      server = new QueryServer(portHolder.getLocalPort(), nettyConfig, null, mock(ChannelHandler.class));
      RuntimeException e = expectThrows(RuntimeException.class, server::start);
      assertTrue(ExceptionUtils.indexOfType(e, IOException.class) >= 0, "Bind failure is not the cause: " + e);
    }

    // The port is free now, so the same server must start
    server.start();
    try {
      assertTrue(connectionOk(server.getChannel().localAddress()));
    } finally {
      server.shutDown();
    }
  }

  @Test
  public void testShutDownWithoutStart() {
    new QueryServer(0, new NettyConfig(), null, mock(ChannelHandler.class)).shutDown();
  }

  @Test
  public void testStartWhileStarted() {
    PinotMetricUtils.init(new PinotConfiguration());
    ServerMetrics.register(new ServerMetrics(PinotMetricUtils.getPinotMetricsRegistry()));
    QueryServer server = new QueryServer(0, new NettyConfig(), null, mock(ChannelHandler.class));
    server.start();
    try {
      expectThrows(IllegalStateException.class, server::start);
    } finally {
      server.shutDown();
    }
  }

  private static boolean connectionOk(InetSocketAddress address) {
    Socket s = null;
    try {
      s = new Socket(address.getHostName(), address.getPort());
      return s.isConnected();
    } catch (Exception e) {
      return false;
    } finally {
      IOUtils.closeQuietly(s);
    }
  }
}

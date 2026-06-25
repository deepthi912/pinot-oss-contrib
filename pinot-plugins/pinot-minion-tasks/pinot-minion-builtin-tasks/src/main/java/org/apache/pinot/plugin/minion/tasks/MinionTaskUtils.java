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
package org.apache.pinot.plugin.minion.tasks;

import com.google.common.base.Preconditions;
import java.net.URI;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TimeZone;
import javax.annotation.Nullable;
import org.apache.commons.lang3.StringUtils;
import org.apache.helix.HelixAdmin;
import org.apache.helix.model.ExternalView;
import org.apache.helix.model.InstanceConfig;
import org.apache.pinot.common.auth.AuthProviderUtils;
import org.apache.pinot.common.auth.NullAuthProvider;
import org.apache.pinot.common.restlet.resources.ValidDocIdsBitmapResponse;
import org.apache.pinot.common.restlet.resources.ValidDocIdsMetadataInfo;
import org.apache.pinot.common.restlet.resources.ValidDocIdsType;
import org.apache.pinot.common.utils.RoaringBitmapUtils;
import org.apache.pinot.common.utils.ServiceStatus;
import org.apache.pinot.common.utils.config.InstanceUtils;
import org.apache.pinot.controller.helix.core.minion.ClusterInfoAccessor;
import org.apache.pinot.controller.util.ServerSegmentMetadataReader;
import org.apache.pinot.core.common.MinionConstants;
import org.apache.pinot.minion.MinionContext;
import org.apache.pinot.spi.auth.AuthProvider;
import org.apache.pinot.spi.config.table.TableConfig;
import org.apache.pinot.spi.config.table.TableTaskConfig;
import org.apache.pinot.spi.config.table.UpsertConfig;
import org.apache.pinot.spi.env.PinotConfiguration;
import org.apache.pinot.spi.filesystem.LocalPinotFS;
import org.apache.pinot.spi.filesystem.PinotFS;
import org.apache.pinot.spi.filesystem.PinotFSFactory;
import org.apache.pinot.spi.ingestion.batch.BatchConfigProperties;
import org.apache.pinot.spi.plugin.PluginManager;
import org.apache.pinot.spi.utils.CommonConstants;
import org.apache.pinot.spi.utils.Enablement;
import org.apache.pinot.spi.utils.IngestionConfigUtils;
import org.apache.pinot.spi.utils.builder.TableNameBuilder;
import org.roaringbitmap.RoaringBitmap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class MinionTaskUtils {
  private static final Logger LOGGER = LoggerFactory.getLogger(MinionTaskUtils.class);

  /**
   * Parses the validDocIdsConsensusMode config string. Defaults to {@code EQUAL} when the value is null or blank.
   * Used by both the executor (bitmap-level consensus) and the task generator (metadata-level consensus).
   */
  public static MinionConstants.ValidDocIdsConsensusMode parseValidDocIdsConsensusMode(String value) {
    if (value == null || value.isBlank()) {
      return MinionConstants.ValidDocIdsConsensusMode.EQUAL;
    }
    return MinionConstants.ValidDocIdsConsensusMode.valueOf(value.toUpperCase().trim());
  }

  private static final String DEFAULT_DIR_PATH_TERMINATOR = "/";

  public static final String DATETIME_PATTERN = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'";
  public static final String UTC = "UTC";

  /**
   * When true, allows METADATA push mode with local FS output dir. Intended for integration tests only.
   * Production should leave this unset (defaults to false); local FS then always uses TAR push.
   */
  public static final String ALLOW_METADATA_PUSH_WITH_LOCAL_FS = "allowMetadataPushWithLocalFs";

  private MinionTaskUtils() {
  }

  /**
   * Resolves the AuthProvider to use for Minion tasks.
   * Priority order:
   * 1. If AUTH_TOKEN is explicitly provided in task configs (by Controller), use it for this specific task
   * 2. Otherwise, fall back to the runtime AuthProvider from MinionContext (enables per-request token rotation)
   *
   * This allows any minion task or util to resolve auth from task configs without requiring callers to pass
   * AuthProvider explicitly.
   */
  public static AuthProvider resolveAuthProvider(Map<String, String> taskConfigs) {
    String explicitToken = taskConfigs.get(MinionConstants.AUTH_TOKEN);
    if (StringUtils.isNotBlank(explicitToken)) {
      return AuthProviderUtils.makeAuthProvider(explicitToken);
    }

    AuthProvider runtimeProvider = MinionContext.getInstance().getTaskAuthProvider();
    if (runtimeProvider == null || runtimeProvider instanceof NullAuthProvider) {
      return new NullAuthProvider();
    }

    return runtimeProvider;
  }

  /**
   * Resolves the auth token string to use for Minion tasks (e.g. for specs that accept a token string).
   * If AUTH_TOKEN is already present in task configs, returns it without creating an AuthProvider.
   * Otherwise resolves via {@link #resolveAuthProvider} and returns its static token.
   *
   * @param taskConfigs task config map (may contain MinionConstants.AUTH_TOKEN)
   * @return auth token string, or null if none
   */
  @Nullable
  public static String resolveAuthToken(Map<String, String> taskConfigs) {
    String explicitToken = taskConfigs.get(MinionConstants.AUTH_TOKEN);
    if (StringUtils.isNotBlank(explicitToken)) {
      return explicitToken;
    }
    return AuthProviderUtils.toStaticToken(resolveAuthProvider(taskConfigs));
  }

  public static PinotFS getInputPinotFS(Map<String, String> taskConfigs, URI fileURI)
      throws Exception {
    String fileURIScheme = fileURI.getScheme();
    if (fileURIScheme == null) {
      return new LocalPinotFS();
    }
    // Try to create PinotFS using given Input FileSystem config always
    String fsClass = taskConfigs.get(BatchConfigProperties.INPUT_FS_CLASS);
    if (fsClass != null) {
      PinotFS pinotFS = PluginManager.get().createInstance(fsClass);
      PinotConfiguration fsProps = IngestionConfigUtils.getInputFsProps(taskConfigs);
      pinotFS.init(fsProps);
      return pinotFS;
    }
    return PinotFSFactory.create(fileURIScheme);
  }

  public static PinotFS getOutputPinotFS(Map<String, String> taskConfigs, URI fileURI)
      throws Exception {
    String fileURIScheme = (fileURI == null) ? null : fileURI.getScheme();
    if (fileURIScheme == null) {
      return new LocalPinotFS();
    }
    // Try to create PinotFS using given Input FileSystem config always
    String fsClass = taskConfigs.get(BatchConfigProperties.OUTPUT_FS_CLASS);
    if (fsClass != null) {
      PinotFS pinotFS = PluginManager.get().createInstance(fsClass);
      PinotConfiguration fsProps = IngestionConfigUtils.getOutputFsProps(taskConfigs);
      pinotFS.init(fsProps);
      return pinotFS;
    }
    return PinotFSFactory.create(fileURIScheme);
  }

  public static URI getOutputSegmentDirURI(Map<String, String> taskConfigs, ClusterInfoAccessor clusterInfoAccessor,
      String tableName) {
    // taskConfigs has priority over clusterInfo configs for output.segment.dir.uri
    String outputDir = taskConfigs.getOrDefault(BatchConfigProperties.OUTPUT_SEGMENT_DIR_URI,
        normalizeDirectoryURI(clusterInfoAccessor.getDataDir()) + TableNameBuilder.extractRawTableName(tableName));
    return URI.create(outputDir);
  }

  public static Map<String, String> getPushTaskConfig(String tableName, Map<String, String> taskConfigs,
      ClusterInfoAccessor clusterInfoAccessor) {
    Map<String, String> singleFileGenerationTaskConfig = new HashMap<>(taskConfigs);
    try {
      String pushMode = IngestionConfigUtils.getPushMode(taskConfigs);

      // Default value for Segment Push Type is TAR.
      BatchConfigProperties.SegmentPushType segmentPushType;
      if (pushMode == null) {
        segmentPushType = BatchConfigProperties.SegmentPushType.TAR;
      } else {
        segmentPushType = BatchConfigProperties.SegmentPushType.valueOf(pushMode.toUpperCase());
      }

      URI outputSegmentDirURI = getOutputSegmentDirURI(taskConfigs, clusterInfoAccessor, tableName);
      if (!isLocalOutputDir(outputSegmentDirURI.getScheme())) {
        switch (segmentPushType) {
          case URI:
            singleFileGenerationTaskConfig.put(BatchConfigProperties.OUTPUT_SEGMENT_DIR_URI,
                outputSegmentDirURI.toString());
            LOGGER.warn("URI push type is not supported in this task. Switching to METADATA push");
            singleFileGenerationTaskConfig.put(BatchConfigProperties.PUSH_MODE,
                BatchConfigProperties.SegmentPushType.METADATA.toString());
            break;
          case METADATA:
            singleFileGenerationTaskConfig.put(BatchConfigProperties.OUTPUT_SEGMENT_DIR_URI,
                outputSegmentDirURI.toString());
            break;
          default:
            singleFileGenerationTaskConfig.put(BatchConfigProperties.PUSH_MODE,
                BatchConfigProperties.SegmentPushType.TAR.toString());
            break;
        }
      } else {
        boolean allowMetadataPushWithLocalFs = Boolean.parseBoolean(
            taskConfigs.getOrDefault(ALLOW_METADATA_PUSH_WITH_LOCAL_FS, "false"));
        if (allowMetadataPushWithLocalFs && pushMode != null) {
          // Override for integration tests: respect explicit push mode with local FS
          singleFileGenerationTaskConfig.put(BatchConfigProperties.OUTPUT_SEGMENT_DIR_URI,
              outputSegmentDirURI.toString());
          singleFileGenerationTaskConfig.put(BatchConfigProperties.PUSH_MODE,
              segmentPushType.toString());
        } else {
          // Production: default to TAR for local output dir
          LOGGER.warn("Local output dir found, defaulting to TAR: {}.", outputSegmentDirURI);
          singleFileGenerationTaskConfig.put(BatchConfigProperties.PUSH_MODE,
              BatchConfigProperties.SegmentPushType.TAR.toString());
        }
      }

      singleFileGenerationTaskConfig.put(BatchConfigProperties.PUSH_CONTROLLER_URI,
          clusterInfoAccessor.getVipUrlForLeadController(tableName));
      return singleFileGenerationTaskConfig;
    } catch (Exception e) {
      singleFileGenerationTaskConfig.put(BatchConfigProperties.PUSH_MODE,
          BatchConfigProperties.SegmentPushType.TAR.toString());
      return singleFileGenerationTaskConfig;
    }
  }

  public static boolean isLocalOutputDir(String outputDirURIScheme) {
    return outputDirURIScheme == null || outputDirURIScheme.startsWith("file");
  }

  public static PinotFS getLocalPinotFs() {
    return new LocalPinotFS();
  }

  public static String normalizeDirectoryURI(URI dirURI) {
    return normalizeDirectoryURI(dirURI.toString());
  }

  public static String normalizeDirectoryURI(String dirInStr) {
    if (!dirInStr.endsWith(DEFAULT_DIR_PATH_TERMINATOR)) {
      return dirInStr + DEFAULT_DIR_PATH_TERMINATOR;
    }
    return dirInStr;
  }

  public static List<String> getServers(String segmentName, String tableNameWithType, HelixAdmin helixAdmin,
      String clusterName) {
    ExternalView externalView = helixAdmin.getResourceExternalView(clusterName, tableNameWithType);
    if (externalView == null) {
      throw new IllegalStateException("External view does not exist for table: " + tableNameWithType);
    }
    Map<String, String> instanceStateMap = externalView.getStateMap(segmentName);
    if (instanceStateMap == null) {
      throw new IllegalStateException("Failed to find segment: " + segmentName);
    }
    ArrayList<String> servers = new ArrayList<>();
    for (Map.Entry<String, String> entry : instanceStateMap.entrySet()) {
      if (entry.getValue().equals(CommonConstants.Helix.StateModel.SegmentStateModel.ONLINE)) {
        servers.add(entry.getKey());
      }
    }
    if (servers.isEmpty()) {
      throw new IllegalStateException("Failed to find any ONLINE servers for segment: " + segmentName);
    }
    return servers;
  }

  /**
   * Extract allowDownloadFromServer config from table task config
   */
  public static boolean extractMinionAllowDownloadFromServer(TableConfig tableConfig, String taskType,
      boolean defaultValue) {
    TableTaskConfig tableTaskConfig = tableConfig.getTaskConfig();
    if (tableTaskConfig != null) {
      Map<String, String> configs = tableTaskConfig.getConfigsForTaskType(taskType);
      if (configs != null && !configs.isEmpty()) {
        return Boolean.parseBoolean(
            configs.getOrDefault(TableTaskConfig.MINION_ALLOW_DOWNLOAD_FROM_SERVER, String.valueOf(defaultValue)));
      }
    }
    return defaultValue;
  }

  /**
   * Per-replica predicate shared by the executor and the task generator: true when the replica's CRC matches the
   * expected CRC. Both values are compared as strings to match the wire-format the server returns.
   */
  public static boolean isReplicaCrcMatching(String expectedCrc, String replicaCrc) {
    return expectedCrc != null && expectedCrc.equals(replicaCrc);
  }

  /**
   * Per-replica predicate shared by the executor and the task generator: true when the replica's reporting server
   * is in a READY state. {@code null} is treated as ready to preserve existing behavior for responses that don't
   * carry a server status.
   */
  public static boolean isReplicaServerReady(@Nullable ServiceStatus.Status status) {
    return status == null || status.equals(ServiceStatus.Status.GOOD);
  }

  /**
   * Cross-replica consensus resolver shared by upsert task generators. Given the per-replica metadata for one
   * segment, returns a single {@link ValidDocIdsMetadataInfo} to drive the per-segment scheduling decision, or
   * {@code null} to skip the segment.
   *
   * - {@code UNSAFE}: returns the first replica with matching CRC and READY server; {@code null} if none.
   * - {@code EQUAL}: requires (a) at least {@code expectedReplicas} responses, (b) every replica has matching
   *   CRC and READY server, (c) all replicas report the same {@code totalDocs}, {@code totalValidDocs}, and
   *   {@code totalInvalidDocs}. Returns the first replica when consistent; {@code null} otherwise.
   * - {@code MOST_VALID_DOCS}: same per-replica requirements as {@code EQUAL} plus the replica-count check,
   *   then returns the replica with the highest {@code totalValidDocs}.
   *
   * {@code expectedReplicas <= 0} disables the replica-count check.
   */
  @Nullable
  public static ValidDocIdsMetadataInfo chooseValidDocIdsReplica(String segmentName, long expectedCrc,
      @Nullable List<ValidDocIdsMetadataInfo> replicas, MinionConstants.ValidDocIdsConsensusMode mode,
      int expectedReplicas) {
    if (replicas == null || replicas.isEmpty()) {
      return null;
    }
    String expectedCrcStr = String.valueOf(expectedCrc);
    if (mode == MinionConstants.ValidDocIdsConsensusMode.UNSAFE) {
      for (ValidDocIdsMetadataInfo replica : replicas) {
        if (isReplicaCrcMatching(expectedCrcStr, replica.getSegmentCrc())
            && isReplicaServerReady(replica.getServerStatus())) {
          return replica;
        }
      }
      return null;
    }
    if (expectedReplicas > 0 && replicas.size() < expectedReplicas) {
      LOGGER.warn("Segment {} returned validDocIds metadata from only {} of {} replicas (consensusMode={}); "
          + "skipping task generation", segmentName, replicas.size(), expectedReplicas, mode);
      return null;
    }
    for (ValidDocIdsMetadataInfo replica : replicas) {
      if (!isReplicaCrcMatching(expectedCrcStr, replica.getSegmentCrc())) {
        LOGGER.warn("CRC mismatch for segment: {} on server {} (zkCrc={}, serverCrc={}); skipping task generation "
            + "(consensusMode={})", segmentName, replica.getInstanceId(), expectedCrcStr, replica.getSegmentCrc(),
            mode);
        return null;
      }
      if (!isReplicaServerReady(replica.getServerStatus())) {
        LOGGER.warn("Server {} is in {} state for segment: {}; skipping task generation (consensusMode={})",
            replica.getInstanceId(), replica.getServerStatus(), segmentName, mode);
        return null;
      }
    }
    if (mode == MinionConstants.ValidDocIdsConsensusMode.EQUAL) {
      ValidDocIdsMetadataInfo first = replicas.get(0);
      for (ValidDocIdsMetadataInfo replica : replicas) {
        if (replica.getTotalDocs() != first.getTotalDocs()
            || replica.getTotalValidDocs() != first.getTotalValidDocs()
            || replica.getTotalInvalidDocs() != first.getTotalInvalidDocs()) {
          LOGGER.warn("Replicas disagree on valid doc counts for segment {} (consensusMode=EQUAL); "
              + "skipping task generation", segmentName);
          return null;
        }
      }
      return first;
    }
    // MOST_VALID_DOCS: pick the replica reporting the most valid docs.
    ValidDocIdsMetadataInfo chosen = replicas.get(0);
    for (ValidDocIdsMetadataInfo replica : replicas) {
      if (replica.getTotalValidDocs() > chosen.getTotalValidDocs()) {
        chosen = replica;
      }
    }
    return chosen;
  }

  /**
   * Returns the validDocIds bitmap from server(s). {@code comparisonMode} is the task config value: UNSAFE,
   * EQUAL (default), or MOST_VALID_DOCS.
   */
  @Nullable
  public static RoaringBitmap getValidDocIdFromServerMatchingCrc(String tableNameWithType, String segmentName,
      String validDocIdsType, MinionContext minionContext, String expectedCrc, String comparisonModeStr) {
    MinionConstants.ValidDocIdsConsensusMode consensusMode = parseValidDocIdsConsensusMode(comparisonModeStr);
    String clusterName = minionContext.getHelixManager().getClusterName();
    HelixAdmin helixAdmin = minionContext.getHelixManager().getClusterManagmentTool();
    List<String> servers = getServers(segmentName, tableNameWithType, helixAdmin, clusterName);
    List<RoaringBitmap> matchingBitmaps = new ArrayList<>();

    for (String server : servers) {
      InstanceConfig instanceConfig = helixAdmin.getInstanceConfig(clusterName, server);
      String endpoint = InstanceUtils.getServerAdminEndpoint(instanceConfig);

      ServerSegmentMetadataReader serverSegmentMetadataReader = new ServerSegmentMetadataReader();
      ValidDocIdsBitmapResponse validDocIdsBitmapResponse;
      try {
        validDocIdsBitmapResponse =
            serverSegmentMetadataReader.getValidDocIdsBitmapFromServer(tableNameWithType, segmentName, endpoint,
                validDocIdsType, 60_000);
      } catch (Exception e) {
        if (consensusMode == MinionConstants.ValidDocIdsConsensusMode.UNSAFE) {
          LOGGER.warn(
              "Unable to retrieve validDocIds bitmap for segment: " + segmentName + " from endpoint: " + endpoint, e);
          continue;
        } else {
          throw new IllegalStateException(
              "Unable to retrieve validDocIds bitmap for segment: " + segmentName + " from endpoint: " + endpoint, e);
        }
      }

      String crcFromValidDocIdsBitmap = validDocIdsBitmapResponse.getSegmentCrc();
      // Check crc from the downloaded segment against the crc returned from the server along with the valid doc id
      // bitmap. If this doesn't match, this means that we are hitting the race condition where the segment has been
      // uploaded successfully while the server is still reloading the segment. Reloading can take a while when the
      // offheap upsert is used because we will need to delete & add all primary keys.
      // `BaseSingleSegmentConversionExecutor.executeTask()` already checks for the crc from the task generator
      // against the crc from the current segment zk metadata, so we don't need to check that here.
      if (!isReplicaCrcMatching(expectedCrc, crcFromValidDocIdsBitmap)) {
        if (consensusMode == MinionConstants.ValidDocIdsConsensusMode.UNSAFE) {
          LOGGER.warn("CRC mismatch for segment: {} from endpoint {}, skipping", segmentName, endpoint);
          continue;
        } else {
          throw new IllegalStateException(
              "CRC mismatch for segment: " + segmentName + ", expected: " + expectedCrc + ", actual from endpoint "
                  + endpoint + ": " + crcFromValidDocIdsBitmap);
        }
      }

      if (!isReplicaServerReady(validDocIdsBitmapResponse.getServerStatus())) {
        if (consensusMode == MinionConstants.ValidDocIdsConsensusMode.UNSAFE) {
          LOGGER.warn("Server {} not READY for segment {}, skipping", validDocIdsBitmapResponse.getInstanceId(),
              segmentName);
          continue;
        } else {
          throw new IllegalStateException("Server " + validDocIdsBitmapResponse.getInstanceId() + " is in "
              + validDocIdsBitmapResponse.getServerStatus() + " state for segment: " + segmentName
              + ". Failing task to avoid inconsistency among replicas.");
        }
      }

      RoaringBitmap bitmap = RoaringBitmapUtils.deserialize(validDocIdsBitmapResponse.getBitmap());
      int cardinality = bitmap.getCardinality();

      if (consensusMode == MinionConstants.ValidDocIdsConsensusMode.UNSAFE) {
        LOGGER.info("Using server {} with {} valid docs for segment {} (mode=UNSAFE)", server, cardinality,
            segmentName);
        return bitmap;
      }

      matchingBitmaps.add(bitmap);
    }

    if (matchingBitmaps.isEmpty()) {
      return null;
    }

    if (consensusMode == MinionConstants.ValidDocIdsConsensusMode.EQUAL) {
      RoaringBitmap consensusBitMap = matchingBitmaps.get(0);
      for (RoaringBitmap b : matchingBitmaps) {
        if (!b.equals(consensusBitMap)) {
          throw new IllegalStateException("No consensus on validDocs across replicas for segment: " + segmentName
              + ". Failing task to avoid replica inconsistency.");
        }
      }
      LOGGER.info("All {} servers have {} valid docs for segment {}", servers.size(), consensusBitMap.getCardinality(),
          segmentName);
      return consensusBitMap;
    }

    // MOST_VALID_DOCS: explicitly pick the bitmap with the maximum valid doc count
    RoaringBitmap maxCardinalityMap = null;
    int maxCard = -1;
    for (RoaringBitmap b : matchingBitmaps) {
      int card = b.getCardinality();
      if (card > maxCard) {
        maxCard = card;
        maxCardinalityMap = b;
      }
    }
    if (maxCardinalityMap != null) {
      LOGGER.info("Selected server with {} valid docs for segment {} (mode=MOST_VALID_DOCS, checked {} servers)",
          maxCard, segmentName, servers.size());
    }
    return maxCardinalityMap;
  }

  public static String toUTCString(long epochMillis) {
    Date date = new Date(epochMillis);
    SimpleDateFormat isoFormat = new SimpleDateFormat(DATETIME_PATTERN);
    isoFormat.setTimeZone(TimeZone.getTimeZone(UTC));
    return isoFormat.format(date);
  }

  public static long fromUTCString(String utcString) {
    return Instant.parse(utcString).toEpochMilli();
  }

  /**
   * Get the validDocIdsType based on the upsertConfig and taskConfigs.
   * The default value is determined by whether delete is enabled in the upsertConfig. If delete is enabled,
   * the default value is 'snapshot_with_delete', otherwise it is 'snapshot'.
   * If delete is enabled, we override the user-specified value to 'snapshot_with_delete' for backward compatibility
   * except when it is 'in_memory_with_delete'.
   * It also validates the combination of validDocIdsType, snapshot and deleteRecordColumn.
   * @param upsertConfig upsertConfig of the table
   * @param taskConfigs taskConfigs of the task
   * @param validDocIdsTypeKey the key to get validDocIdsType from taskConfigs
   * @return the validDocIdsType
   */
  public static ValidDocIdsType getValidDocIdsType(UpsertConfig upsertConfig, Map<String, String> taskConfigs,
      String validDocIdsTypeKey) {
    boolean isDeleteEnabled = StringUtils.isNotEmpty(upsertConfig.getDeleteRecordColumn());
    ValidDocIdsType defaultValidDocIdsType =
        isDeleteEnabled ? ValidDocIdsType.SNAPSHOT_WITH_DELETE : ValidDocIdsType.SNAPSHOT;
    String validDocIdsTypeStr = taskConfigs.getOrDefault(validDocIdsTypeKey,
        defaultValidDocIdsType.name()).toUpperCase();
    ValidDocIdsType validDocIdsType = ValidDocIdsType.valueOf(validDocIdsTypeStr);

    if (isDeleteEnabled && validDocIdsType != ValidDocIdsType.SNAPSHOT_WITH_DELETE
        && validDocIdsType != ValidDocIdsType.IN_MEMORY_WITH_DELETE) {
      LOGGER.warn(
          "Overriding user-specified validDocIdsType '{}' to '{}' for backward compatibility because delete is "
              + "enabled (deleteRecordColumn='{}').",
          validDocIdsType, ValidDocIdsType.SNAPSHOT_WITH_DELETE, upsertConfig.getDeleteRecordColumn());
      validDocIdsType = ValidDocIdsType.SNAPSHOT_WITH_DELETE;
    }

    if (validDocIdsType == ValidDocIdsType.SNAPSHOT || validDocIdsType == ValidDocIdsType.SNAPSHOT_WITH_DELETE) {
      Preconditions.checkState(upsertConfig.getSnapshot() != Enablement.DISABLE,
          "'snapshot' must not be 'DISABLE' with validDocIdsType: %s", validDocIdsType);
    }

    if (validDocIdsType == ValidDocIdsType.IN_MEMORY_WITH_DELETE
        || validDocIdsType == ValidDocIdsType.SNAPSHOT_WITH_DELETE) {
      Preconditions.checkState(isDeleteEnabled,
          "'deleteRecordColumn' must be provided with validDocIdsType: %s", validDocIdsType);
    }
    return validDocIdsType;
  }
}

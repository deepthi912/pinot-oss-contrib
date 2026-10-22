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
// Experimental Metal (Apple GPU) kernels for Pinot server-side GROUP BY over dictionary-encoded columns.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unistd.h>

static const char* kShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Params {
  uint numRows;       // rows in this dispatch
  uint numGroups;
  uint hasFilter;
  uint rowsPerSlice;  // rows owned by one threadgroup (local kernel only)
  uint rowOffset;     // first row of this dispatch (global kernel only)
  uint hasValues;     // 0 = COUNT only, no value column bound
  uint hasKeys;       // 0 = no GROUP BY: every row is group 0
};

struct DistinctParams {
  uint numRows;
  uint numGroups;
  uint cardinality;
  uint hasFilter;
  uint hasKeys;
};

struct FoldParams {
  uint numGroups;
  uint numSlices;
  uint numParts;      // 3 = (lo16, hi16, count), 5 = (b0, b1, b2, b3, count)
  uint clear;         // zero the partials after folding
};

constant uint LOCAL_MAX_GROUPS = 1024;

inline bool passes(device const uint* filter, uint hasFilter, uint r) {
  return hasFilter == 0 || ((filter[r >> 5] >> (r & 31)) & 1u) != 0;
}

// Low cardinality: per-threadgroup atomics over <= 65536 rows; 16-bit halves cannot overflow (65536 * 65535 < 2^32).
kernel void gb_local(device const int* dictIds [[buffer(0)]],
                     device const int* values [[buffer(1)]],
                     device const uint* filter [[buffer(2)]],
                     device uint* partials [[buffer(3)]],
                     constant Params& p [[buffer(4)]],
                     uint tgid [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint tgsize [[threads_per_threadgroup]]) {
  threadgroup atomic_uint acc[3 * LOCAL_MAX_GROUPS];
  uint G = p.numGroups;
  uint n = 3 * G;
  for (uint i = tid; i < n; i += tgsize) {
    atomic_store_explicit(&acc[i], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint start = tgid * p.rowsPerSlice;
  uint end = min(start + p.rowsPerSlice, p.numRows);
  // Without group keys every row hits one address: sum in registers, then one atomic per SIMD group.
  uint lo = 0, hi = 0, count = 0;
  for (uint r = start + tid; r < end; r += tgsize) {
    if (!passes(filter, p.hasFilter, r)) continue;
    if (!p.hasKeys) {
      if (p.hasValues) {
        uint u = as_type<uint>(values[r]) ^ 0x80000000u;
        lo += u & 0xFFFFu;
        hi += u >> 16;
      }
      count++;
      continue;
    }
    uint g = (uint) dictIds[r];
    if (g >= G) continue;
    if (p.hasValues) {
      uint u = as_type<uint>(values[r]) ^ 0x80000000u;
      atomic_fetch_add_explicit(&acc[g], u & 0xFFFFu, memory_order_relaxed);
      atomic_fetch_add_explicit(&acc[G + g], u >> 16, memory_order_relaxed);
    }
    atomic_fetch_add_explicit(&acc[2 * G + g], 1u, memory_order_relaxed);
  }
  if (!p.hasKeys) {
    lo = simd_sum(lo);
    hi = simd_sum(hi);
    count = simd_sum(count);
    if (simd_is_first()) {
      atomic_fetch_add_explicit(&acc[0], lo, memory_order_relaxed);
      atomic_fetch_add_explicit(&acc[G], hi, memory_order_relaxed);
      atomic_fetch_add_explicit(&acc[2 * G], count, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device uint* out = partials + tgid * n;
  for (uint i = tid; i < n; i += tgsize) {
    out[i] = atomic_load_explicit(&acc[i], memory_order_relaxed);
  }
}

// High cardinality: device-memory atomics on byte pieces, folded every 2^24 rows (255 * 2^24 < 2^32).
kernel void gb_global(device const int* dictIds [[buffer(0)]],
                      device const int* values [[buffer(1)]],
                      device const uint* filter [[buffer(2)]],
                      device atomic_uint* partials [[buffer(3)]],
                      constant Params& p [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
  if (gid >= p.numRows) return;
  uint r = gid + p.rowOffset;
  if (!passes(filter, p.hasFilter, r)) return;
  uint G = p.numGroups;
  uint g = p.hasKeys ? (uint) dictIds[r] : 0u;
  if (g >= G) return;
  if (p.hasValues) {
    uint u = as_type<uint>(values[r]) ^ 0x80000000u;
    uint b0 = u & 0xFFu, b1 = (u >> 8) & 0xFFu, b2 = (u >> 16) & 0xFFu, b3 = u >> 24;
    if (b0) atomic_fetch_add_explicit(&partials[g], b0, memory_order_relaxed);
    if (b1) atomic_fetch_add_explicit(&partials[G + g], b1, memory_order_relaxed);
    if (b2) atomic_fetch_add_explicit(&partials[2 * G + g], b2, memory_order_relaxed);
    if (b3) atomic_fetch_add_explicit(&partials[3 * G + g], b3, memory_order_relaxed);
  }
  atomic_fetch_add_explicit(&partials[4 * G + g], 1u, memory_order_relaxed);
}

// One thread per group: combine all slices of narrow partial sums into 64-bit accumulators.
kernel void gb_fold(device uint* partials [[buffer(0)]],
                    device ulong* accSum [[buffer(1)]],
                    device ulong* accCount [[buffer(2)]],
                    constant FoldParams& fp [[buffer(3)]],
                    uint g [[thread_position_in_grid]]) {
  uint G = fp.numGroups;
  if (g >= G) return;
  ulong s = 0, c = 0;
  for (uint sl = 0; sl < fp.numSlices; sl++) {
    device uint* base = partials + (ulong) sl * fp.numParts * G;
    if (fp.numParts == 3) {
      s += (ulong) base[g] + ((ulong) base[G + g] << 16);
      c += base[2 * G + g];
    } else {
      s += (ulong) base[g] + ((ulong) base[G + g] << 8) + ((ulong) base[2 * G + g] << 16)
          + ((ulong) base[3 * G + g] << 24);
      c += base[4 * G + g];
    }
    if (fp.clear) {
      for (uint k = 0; k < fp.numParts; k++) base[k * G + g] = 0u;
    }
  }
  accSum[g] += s;
  accCount[g] += c;
}

// Exact DISTINCTCOUNT: sets bit (group * cardinality + id) for every matching row.
kernel void distinct_pairs(device const int* keys [[buffer(0)]],
                           device const int* ids [[buffer(1)]],
                           device const uint* filter [[buffer(2)]],
                           device atomic_uint* bits [[buffer(3)]],
                           constant DistinctParams& p [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid >= p.numRows) return;
  if (!passes(filter, p.hasFilter, gid)) return;
  uint g = p.hasKeys ? (uint) keys[gid] : 0u;
  uint id = (uint) ids[gid];
  if (g >= p.numGroups || id >= p.cardinality) return;
  uint bit = g * p.cardinality + id;
  uint mask = 1u << (bit & 31);
  device atomic_uint* word = &bits[bit >> 5];
  if ((atomic_load_explicit(word, memory_order_relaxed) & mask) == 0) {
    atomic_fetch_or_explicit(word, mask, memory_order_relaxed);
  }
}
)METAL";

namespace {

struct Params {
  uint32_t numRows;
  uint32_t numGroups;
  uint32_t hasFilter;
  uint32_t rowsPerSlice;
  uint32_t rowOffset;
  uint32_t hasValues;
  uint32_t hasKeys;
};

struct DistinctParams {
  uint32_t numRows;
  uint32_t numGroups;
  uint32_t cardinality;
  uint32_t hasFilter;
  uint32_t hasKeys;
};

constexpr uint64_t kMaxDistinctBits = 1ull << 28;

struct FoldParams {
  uint32_t numGroups;
  uint32_t numSlices;
  uint32_t numParts;
  uint32_t clear;
};

constexpr uint32_t kLocalMaxGroups = 1024;
constexpr uint32_t kMaxRowsPerSlice = 65536;
constexpr uint32_t kGlobalBatchRows = 1u << 24;
constexpr NSUInteger kThreadsPerGroup = 256;

struct Context {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> gbLocal;
  id<MTLComputePipelineState> gbGlobal;
  id<MTLComputePipelineState> gbFold;
  id<MTLComputePipelineState> distinctPairs;
  id<MTLBuffer> empty;  // bound when there is no filter bitmap
  char deviceName[256];
};

void setError(char* err, int32_t errLen, NSString* message) {
  if (err == nullptr || errLen <= 0) return;
  snprintf(err, (size_t) errLen, "%s", message.UTF8String);
}

// Wraps page-aligned caller memory without copying (unified memory); otherwise copies.
id<MTLBuffer> wrap(Context* ctx, const void* ptr, int64_t bytes) {
  static const uintptr_t pageSize = (uintptr_t) getpagesize();
  if (ptr == nullptr || bytes <= 0) {
    return ctx->empty;
  }
  if (((uintptr_t) ptr % pageSize) == 0 && ((uintptr_t) bytes % pageSize) == 0) {
    return [ctx->device newBufferWithBytesNoCopy:(void*) ptr
                                          length:(NSUInteger) bytes
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
  }
  return [ctx->device newBufferWithBytes:ptr length:(NSUInteger) bytes options:MTLResourceStorageModeShared];
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name, NSError** error) {
  id<MTLFunction> fn = [library newFunctionWithName:name];
  if (fn == nil) return nil;
  return [device newComputePipelineStateWithFunction:fn error:error];
}

bool runAndWait(id<MTLCommandBuffer> cb, double* gpuMs, char* err, int32_t errLen) {
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.status != MTLCommandBufferStatusCompleted) {
    setError(err, errLen, cb.error ? cb.error.localizedDescription : @"command buffer failed");
    return false;
  }
  if (gpuMs != nullptr) *gpuMs = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
  return true;
}

}  // namespace

extern "C" {

void* mg_create(char* err, int32_t errLen) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      setError(err, errLen, @"no Metal device available");
      return nullptr;
    }
    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    id<MTLLibrary> library = [device newLibraryWithSource:@(kShaderSource) options:options error:&error];
    if (library == nil) {
      setError(err, errLen, error ? error.localizedDescription : @"shader compilation failed");
      return nullptr;
    }
    Context* ctx = new Context();
    ctx->device = device;
    ctx->queue = [device newCommandQueue];
    ctx->empty = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    ctx->gbLocal = pipeline(device, library, @"gb_local", &error);
    ctx->gbGlobal = pipeline(device, library, @"gb_global", &error);
    ctx->gbFold = pipeline(device, library, @"gb_fold", &error);
    ctx->distinctPairs = pipeline(device, library, @"distinct_pairs", &error);
    if (ctx->gbLocal == nil || ctx->gbGlobal == nil || ctx->gbFold == nil || ctx->distinctPairs == nil) {
      setError(err, errLen, error ? error.localizedDescription : @"pipeline creation failed");
      delete ctx;
      return nullptr;
    }
    snprintf(ctx->deviceName, sizeof(ctx->deviceName), "%s", device.name.UTF8String);
    return ctx;
  }
}

void mg_destroy(void* handle) {
  @autoreleasepool {
    delete static_cast<Context*>(handle);
  }
}

const char* mg_device_name(void* handle) {
  return static_cast<Context*>(handle)->deviceName;
}

// Registers memory with the GPU once and returns a retained handle; release with mg_unpin.
void* mg_pin(void* handle, const void* ptr, int64_t bytes) {
  @autoreleasepool {
    id<MTLBuffer> buffer = wrap(static_cast<Context*>(handle), ptr, bytes);
    return buffer == nil ? nullptr : (void*) CFBridgingRetain(buffer);
  }
}

void mg_unpin(void* buffer) {
  if (buffer != nullptr) {
    CFBridgingRelease(buffer);
  }
}

// GROUP BY dictId: COUNT(*), plus SUM(value) when valuePin is set; sums are of (value + 2^31) per group.
int32_t mg_group_by(void* handle, void* dictPin, void* valuePin, void* filterPin, int32_t numRows,
    int32_t numGroups, uint64_t* outSumU, uint64_t* outCount, double* gpuMs, char* err, int32_t errLen) {
  @autoreleasepool {
    Context* ctx = static_cast<Context*>(handle);
    if (numRows < 0 || numGroups <= 0) {
      setError(err, errLen, @"numRows must be >= 0 and numGroups > 0");
      return -1;
    }
    uint32_t rows = (uint32_t) numRows;
    uint32_t groups = (uint32_t) numGroups;
    id<MTLBuffer> dictBuf = dictPin != nullptr ? (__bridge id<MTLBuffer>) dictPin : ctx->empty;
    id<MTLBuffer> valueBuf = valuePin != nullptr ? (__bridge id<MTLBuffer>) valuePin : ctx->empty;
    id<MTLBuffer> filterBuf = filterPin != nullptr ? (__bridge id<MTLBuffer>) filterPin : ctx->empty;
    id<MTLBuffer> accSum = [ctx->device newBufferWithLength:groups * sizeof(uint64_t)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> accCount = [ctx->device newBufferWithLength:groups * sizeof(uint64_t)
                                                      options:MTLResourceStorageModeShared];
    if (accSum == nil || accCount == nil) {
      setError(err, errLen, @"GPU buffer allocation failed");
      return -1;
    }
    memset(accSum.contents, 0, accSum.length);
    memset(accCount.contents, 0, accCount.length);

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];  // serial dispatch: each dispatch sees the last
    Params p = {rows, groups, filterPin != nullptr ? 1u : 0u, 0, 0, valuePin != nullptr ? 1u : 0u,
                dictPin != nullptr ? 1u : 0u};
    id<MTLBuffer> partials;

    if (rows > 0 && groups <= kLocalMaxGroups) {
      uint32_t perSlice = (rows + 511) / 512;  // aim for ~512 threadgroups to fill the GPU
      perSlice = perSlice < 1024 ? 1024 : (perSlice > kMaxRowsPerSlice ? kMaxRowsPerSlice : perSlice);
      uint32_t slices = (rows + perSlice - 1) / perSlice;
      p.rowsPerSlice = perSlice;
      partials = [ctx->device newBufferWithLength:(NSUInteger) slices * 3 * groups * sizeof(uint32_t)
                                          options:MTLResourceStorageModePrivate];
      if (partials == nil) {
        [enc endEncoding];
        setError(err, errLen, @"GPU buffer allocation failed");
        return -1;
      }
      [enc setComputePipelineState:ctx->gbLocal];
      [enc setBuffer:dictBuf offset:0 atIndex:0];
      [enc setBuffer:valueBuf offset:0 atIndex:1];
      [enc setBuffer:filterBuf offset:0 atIndex:2];
      [enc setBuffer:partials offset:0 atIndex:3];
      [enc setBytes:&p length:sizeof(p) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(slices, 1, 1) threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];

      FoldParams fp = {groups, slices, 3, 0};
      [enc setComputePipelineState:ctx->gbFold];
      [enc setBuffer:partials offset:0 atIndex:0];
      [enc setBuffer:accSum offset:0 atIndex:1];
      [enc setBuffer:accCount offset:0 atIndex:2];
      [enc setBytes:&fp length:sizeof(fp) atIndex:3];
      [enc dispatchThreads:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];
    } else if (rows > 0) {
      partials = [ctx->device newBufferWithLength:(NSUInteger) 5 * groups * sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];
      if (partials == nil) {
        [enc endEncoding];
        setError(err, errLen, @"GPU buffer allocation failed");
        return -1;
      }
      memset(partials.contents, 0, partials.length);
      FoldParams fp = {groups, 1, 5, 1};
      for (uint32_t offset = 0; offset < rows; offset += kGlobalBatchRows) {
        uint32_t batch = rows - offset < kGlobalBatchRows ? rows - offset : kGlobalBatchRows;
        p.numRows = batch;
        p.rowOffset = offset;
        [enc setComputePipelineState:ctx->gbGlobal];
        [enc setBuffer:dictBuf offset:0 atIndex:0];
        [enc setBuffer:valueBuf offset:0 atIndex:1];
        [enc setBuffer:filterBuf offset:0 atIndex:2];
        [enc setBuffer:partials offset:0 atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];

        [enc setComputePipelineState:ctx->gbFold];
        [enc setBuffer:partials offset:0 atIndex:0];
        [enc setBuffer:accSum offset:0 atIndex:1];
        [enc setBuffer:accCount offset:0 atIndex:2];
        [enc setBytes:&fp length:sizeof(fp) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];
      }
    }
    [enc endEncoding];
    if (!runAndWait(cb, gpuMs, err, errLen)) return -1;
    if (outSumU != nullptr) {
      memcpy(outSumU, accSum.contents, groups * sizeof(uint64_t));
    }
    memcpy(outCount, accCount.contents, groups * sizeof(uint64_t));
    return 0;
  }
}

// Exact DISTINCTCOUNT pairs: outWords receives the (group * cardinality + id) bitmap, ceil(groups * cardinality / 32)
// words. keyPin may be null (single group).
int32_t mg_distinct_pairs(void* handle, void* keyPin, void* idPin, void* filterPin, int32_t numRows, int32_t numGroups,
    int32_t cardinality, void* outPin, double* gpuMs, char* err, int32_t errLen) {
  @autoreleasepool {
    Context* ctx = static_cast<Context*>(handle);
    uint64_t numBits = (uint64_t) numGroups * (uint64_t) cardinality;
    if (numRows < 0 || numGroups <= 0 || cardinality <= 0 || numBits > kMaxDistinctBits) {
      setError(err, errLen, @"invalid distinct dimensions");
      return -1;
    }
    NSUInteger words = (NSUInteger) ((numBits + 31) / 32);
    id<MTLBuffer> bits = (__bridge id<MTLBuffer>) outPin;
    if (bits == nil || bits.length < words * sizeof(uint32_t)) {
      setError(err, errLen, @"distinct output buffer too small");
      return -1;
    }
    memset(bits.contents, 0, words * sizeof(uint32_t));
    if (gpuMs != nullptr) *gpuMs = 0;
    if (numRows > 0) {
      DistinctParams p = {(uint32_t) numRows, (uint32_t) numGroups, (uint32_t) cardinality,
                          filterPin != nullptr ? 1u : 0u, keyPin != nullptr ? 1u : 0u};
      id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx->distinctPairs];
      [enc setBuffer:(keyPin != nullptr ? (__bridge id<MTLBuffer>) keyPin : ctx->empty) offset:0 atIndex:0];
      [enc setBuffer:(__bridge id<MTLBuffer>) idPin offset:0 atIndex:1];
      [enc setBuffer:(filterPin != nullptr ? (__bridge id<MTLBuffer>) filterPin : ctx->empty) offset:0 atIndex:2];
      [enc setBuffer:bits offset:0 atIndex:3];
      [enc setBytes:&p length:sizeof(p) atIndex:4];
      [enc dispatchThreads:MTLSizeMake((NSUInteger) numRows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];
      [enc endEncoding];
      if (!runAndWait(cb, gpuMs, err, errLen)) return -1;
    }
    return 0;
  }
}

}  // extern "C"

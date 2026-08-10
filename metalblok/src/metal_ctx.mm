#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_ctx.hpp"
#include <unistd.h>
#include <cstdio>
#include <cstdlib>

namespace blade {

#define DEV ((__bridge id<MTLDevice>)device_)
#define QUE ((__bridge id<MTLCommandQueue>)queue_)
#define LIB ((__bridge id<MTLLibrary>)library_)
#define CB  ((__bridge id<MTLCommandBuffer>)cmdbuf_)
#define ENC ((__bridge id<MTLComputeCommandEncoder>)encoder_)
#define PSOPTR ((CFMutableDictionaryRef)pso_map_)
#define PSONAMED ((__bridge NSMutableDictionary*)pso_named_)

[[noreturn]] static void die(const char* m) {
    std::fprintf(stderr, "metal: %s\n", m); std::abort();
}

void Metal::init(const char* kernel_path) {
    device_ = (__bridge_retained void*)MTLCreateSystemDefaultDevice();
    if (!device_) die("no Metal device");
    queue_ = (__bridge_retained void*)[DEV newCommandQueue];
    NSError* err = nil;
    NSString* path = [NSString stringWithUTF8String:kernel_path];
#if METALBLOK_KERNEL_IS_SOURCE
    NSString* source = [NSString stringWithContentsOfFile:path
                                                 encoding:NSUTF8StringEncoding
                                                    error:&err];
    if (!source) {
        std::fprintf(stderr, "metal source: %s\n",
                     err.localizedDescription.UTF8String);
        die("read kernels.metal");
    }
    MTLCompileOptions* opts = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeSafe;
    } else {
        opts.fastMathEnabled = NO;
    }
    library_ = (__bridge_retained void*)[DEV newLibraryWithSource:source
                                                          options:opts
                                                            error:&err];
#else
    NSURL* url = [NSURL fileURLWithPath:path];
    library_ = (__bridge_retained void*)[DEV newLibraryWithURL:url error:&err];
#endif
    if (!library_) {
        std::fprintf(stderr, "metal library: %s\n",
                     err.localizedDescription.UTF8String);
        die("compile/load kernels");
    }
    // Pointer-keyed PSO cache: keys are const char* string-literal pointers,
    // values are CFRetain'd id<MTLComputePipelineState>.  No NSString alloc on
    // the hot path.
    pso_map_   = (void*)CFDictionaryCreateMutable(NULL, 0, NULL, &kCFTypeDictionaryValueCallBacks);
    pso_named_ = (__bridge_retained void*)[NSMutableDictionary new];
}

MtlBuf Metal::alloc(size_t bytes) {
    id<MTLBuffer> b = [DEV newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) die("alloc");
    return { (__bridge_retained void*)b, bytes, b.contents, 0 };
}

MtlBuf Metal::wrap(const void* ptr, size_t bytes) {
    // Wrap host memory as GPU buffer.  newBufferWithBytesNoCopy REQUIRES the
    // pointer be page-aligned (16 KiB on Apple Silicon).  Tensors mmap'd from
    // safetensors live at arbitrary offsets within their shard, so we wrap
    // starting from the enclosing page boundary and remember the in-page
    // offset, which we apply at every setBuffer: call site.
    static const size_t PG = (size_t)::getpagesize();
    uintptr_t  raw = (uintptr_t)ptr;
    uintptr_t  pg  = raw & ~(uintptr_t)(PG - 1);
    size_t     off = (size_t)(raw - pg);
    size_t     len = ((off + bytes) + PG - 1) & ~(PG - 1);
    id<MTLBuffer> b = [DEV newBufferWithBytesNoCopy:(void*)pg
                                             length:len
                                            options:MTLResourceStorageModeShared
                                        deallocator:nil];
    if (!b) die("wrap");
    return { (__bridge_retained void*)b, bytes, (void*)ptr, off };
}

void Metal::release(MtlBuf& b) {
    if (b.obj) { id<MTLBuffer> x = (__bridge_transfer id<MTLBuffer>)b.obj; (void)x; b.obj = nullptr; }
}

static id<MTLComputePipelineState> get_pso(id<MTLDevice> dev, id<MTLLibrary> lib,
                                           CFMutableDictionaryRef ptr_cache,
                                           NSMutableDictionary* name_cache,
                                           const char* name) {
    // Hot path: pointer-identity lookup.  Kernel names at all call sites are
    // string literals, so the same logical name always presents the same
    // pointer within a translation unit.
    const void* v = CFDictionaryGetValue(ptr_cache, name);
    if (v) return (__bridge id<MTLComputePipelineState>)v;
    // Cold path: fall back to string-keyed lookup so we never compile the same
    // function twice even if two TUs hand us distinct literal pointers.
    NSString* k = [NSString stringWithUTF8String:name];
    id pso = name_cache[k];
    if (!pso) {
        id<MTLFunction> fn = [lib newFunctionWithName:k];
        if (!fn) { std::fprintf(stderr, "no kernel %s\n", name); std::abort(); }
        NSError* err = nil;
        pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { std::fprintf(stderr, "pso %s: %s\n", name, err.localizedDescription.UTF8String); std::abort(); }
        name_cache[k] = pso;
    }
    CFDictionarySetValue(ptr_cache, name, (__bridge const void*)pso);
    return pso;
}

void Metal::begin() {
    cmdbuf_  = (__bridge_retained void*)[QUE commandBuffer];
    encoder_ = (__bridge_retained void*)[CB computeCommandEncoder];
}

void Metal::commit_and_wait() {
    [ENC endEncoding];
    [CB commit];
    [CB waitUntilCompleted];
    // CRITICAL: a faulted GPU command buffer left unchecked leaves the GPU
    // context wedged -- on Apple Silicon that can hang or panic the whole
    // machine.  Detect it here and abort the process cleanly so the OS reaps
    // the GPU context.  The error names the kernel that was executing.
    if ([CB status] == MTLCommandBufferStatusError || [CB error] != nil) {
        NSError* e = [CB error];
        std::fprintf(stderr,
            "metal: GPU command buffer FAULTED in kernel '%s': %s (code=%ld)\n",
            last_kernel_ ? last_kernel_ : "(unknown)",
            e ? e.localizedDescription.UTF8String : "(no description)",
            e ? (long)e.code : -1L);
        std::abort();
    }
    double gpu_s = [CB GPUEndTime] - [CB GPUStartTime];
    last_gpu_us_  = (long long)(gpu_s * 1e6);
    step_gpu_us  += last_gpu_us_;
    step_cmdbufs += 1;
    { id<MTLComputeCommandEncoder> e = (__bridge_transfer id<MTLComputeCommandEncoder>)encoder_; (void)e; encoder_ = nullptr; }
    { id<MTLCommandBuffer> c = (__bridge_transfer id<MTLCommandBuffer>)cmdbuf_; (void)c; cmdbuf_ = nullptr; }
}

void Metal::flush() { commit_and_wait(); begin(); }

void Metal::dispatch(const char* name,
                     const std::vector<MtlBuf>& bufs,
                     const std::vector<ByteArg>& byte_args,
                     uint32_t grid_x, uint32_t tg_x,
                     bool one_tg_per_grid_x)
{
    id<MTLComputePipelineState> pso = get_pso(DEV, LIB, PSOPTR, PSONAMED, name);
    [ENC setComputePipelineState:pso];
    last_kernel_ = name;
    NSUInteger idx = 0;
    for (const auto& b : bufs) {
        [ENC setBuffer:(__bridge id<MTLBuffer>)b.obj offset:b.offset atIndex:idx++];
    }
    for (const auto& a : byte_args) {
        [ENC setBytes:a.p length:a.n atIndex:idx++];
    }
    MTLSize tgsize = MTLSizeMake(tg_x, 1, 1);
    MTLSize grid   = MTLSizeMake(grid_x, 1, 1);
    // one_tg_per_grid_x distinguishes the two Metal dispatch entry points:
    //   true  -> grid_x is the number of THREADGROUPS to launch
    //   false -> grid_x is the total number of THREADS  (Metal does the divide)
    if (one_tg_per_grid_x) [ENC dispatchThreadgroups:grid threadsPerThreadgroup:tgsize];
    else                   [ENC dispatchThreads:grid threadsPerThreadgroup:tgsize];
}

void Metal::dispatch2d(const char* name,
                       const std::vector<MtlBuf>& bufs,
                       const std::vector<ByteArg>& byte_args,
                       uint32_t grid_x, uint32_t grid_y, uint32_t tg_x)
{
    id<MTLComputePipelineState> pso = get_pso(DEV, LIB, PSOPTR, PSONAMED, name);
    [ENC setComputePipelineState:pso];
    last_kernel_ = name;
    NSUInteger idx = 0;
    for (const auto& b : bufs) {
        [ENC setBuffer:(__bridge id<MTLBuffer>)b.obj offset:b.offset atIndex:idx++];
    }
    for (const auto& a : byte_args) {
        [ENC setBytes:a.p length:a.n atIndex:idx++];
    }
    MTLSize tgsize = MTLSizeMake(tg_x, 1, 1);
    MTLSize grid   = MTLSizeMake(grid_x, grid_y, 1);
    [ENC dispatchThreadgroups:grid threadsPerThreadgroup:tgsize];
}

} // namespace blade

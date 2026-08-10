// Thin Metal context. Holds device, queue, and a name->PSO cache loaded
// once from the embedded metallib. All buffers are MTLResourceStorageModeShared
// (unified memory, zero-copy CPU/GPU). Pointers from mmap'd files are wrapped
// with newBufferWithBytesNoCopy:length:options:deallocator: -- this gives the
// GPU direct access to the page-cached weight bytes with no copy whatsoever.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace blade {

struct MtlBuf {           // opaque; .mm casts back to id<MTLBuffer>
    void*  obj = nullptr;
    size_t length = 0;          // bytes the kernel sees, from `contents`
    void*  contents = nullptr;  // logical CPU pointer to first byte of data
    size_t offset = 0;          // in-page byte offset to apply at setBuffer:
};

class Metal {
public:
    // `kernel_path` is either kernels.metal (default correctness build) or an
    // offline metallib. Runtime source compilation avoids a hard dependency
    // on Xcode's optional command-line Metal component.
    void init(const char* kernel_path);

    // Allocate a private (managed) shared-storage buffer (CPU+GPU visible).
    MtlBuf alloc(size_t bytes);

    // Wrap an existing host pointer as a GPU buffer (no copy). The pointer
    // must remain valid for the lifetime of the buffer (we use mmap regions).
    MtlBuf wrap(const void* ptr, size_t bytes);

    void release(MtlBuf& b);

    // Dispatch a kernel by name. `bufs` are the MTLBuffer args in order.
    // `byte_args` are appended as setBytes at indices starting after bufs.
    // grid_x = total threads in X (we use threadgroup_in_grid for GEMV-style),
    // tg_x   = threads per threadgroup. one_tg_per_grid_x: if true, dispatches
    // grid_x threadgroups of tg_x threads each (matches kernels that use
    // threadgroup_position_in_grid for the row index).
    struct ByteArg { const void* p; size_t n; };
    void dispatch(const char* name,
                  const std::vector<MtlBuf>& bufs,
                  const std::vector<ByteArg>& byte_args,
                  uint32_t grid_x, uint32_t tg_x,
                  bool one_tg_per_grid_x);

    // 2D variant: dispatches grid_x*grid_y THREADGROUPS of tg_x threads each.
    // Threadgroup id is read as uint2 [[threadgroup_position_in_grid]].
    // Used by tiled matmul kernels (output tile = 2D).
    void dispatch2d(const char* name,
                    const std::vector<MtlBuf>& bufs,
                    const std::vector<ByteArg>& byte_args,
                    uint32_t grid_x, uint32_t grid_y, uint32_t tg_x);

    // Begin/commit a single command buffer that batches many dispatches.
    // The runtime uses one cmdbuf per token to maximize GPU pipelining.
    void begin();
    void commit_and_wait();
    // Commit + wait + begin a fresh encoder.  Used to make GPU results
    // (e.g. router top-k indices in a Shared MTLBuffer) host-visible mid-step.
    void flush();

    // Microseconds the GPU spent executing the most-recently-completed
    // command buffer (GPUEndTime - GPUStartTime).  Excludes encoding, queue
    // wait, and CPU/host sync.  Use to attribute step time correctly.
    long long last_gpu_us() const { return last_gpu_us_; }
    // Per-token accumulator.  Reset by reset_step_stats(), incremented by
    // every commit_and_wait() inside the step.  Lets us split total step
    // time into "GPU compute" vs "host sync stall" without touching every
    // kernel call site.
    long long step_gpu_us  = 0;
    int       step_cmdbufs = 0;
    void reset_step_stats() { step_gpu_us = 0; step_cmdbufs = 0; }

private:
    void* device_   = nullptr;     // id<MTLDevice>
    void* queue_    = nullptr;     // id<MTLCommandQueue>
    void* library_  = nullptr;     // id<MTLLibrary>
    void* cmdbuf_   = nullptr;     // id<MTLCommandBuffer>
    void* encoder_  = nullptr;     // id<MTLComputeCommandEncoder>
    // Kernel-name -> id<MTLComputePipelineState>, lazy-built.  All call sites
    // pass string literals, so we key by the literal's pointer identity for an
    // alloc-free O(1) lookup on the per-token critical path.  Falls back to
    // string-keyed lookup on cold-miss (handles the rare case of duplicate
    // literals from different translation units).
    void* pso_map_  = nullptr;     // CFMutableDictionaryRef (pointer-keyed)
    void* pso_named_ = nullptr;    // NSMutableDictionary*  (string-keyed cold path)
    long long last_gpu_us_  = 0;
    // Name of the most recently encoded kernel in the current command buffer.
    // Used only to make a GPU command-buffer fault report which kernel was
    // executing, so a fault aborts cleanly instead of hanging the GPU/host.
    const char* last_kernel_ = nullptr;
};

} // namespace blade

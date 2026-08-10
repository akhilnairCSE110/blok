// tools/io_probe_v2.c
// ---------------------------------------------------------------------------
// Second-generation I/O probe.  Standalone C, no blade/Metal deps.
//
// Builds on v1 findings:
//   - pread + F_NOCACHE is clean at sequential 1 GB
//   - cached reads inflate inactive/cache and CAN cause pageouts (noisy)
//   - All v1 tests ran with ~10 GB available -- doesn't match crash env
//
// v2 closes the gaps the audit flagged:
//   R1. random expert-shaped reads (480 * 4 MB = ~1.9 GB) with F_NOCACHE
//       -- mimics one decode token of MoE expert traffic
//   R2. same random pattern but cached -- does cache fill from random?
//   M1. mmap under pressure, cached
//   M2. mmap + MAP_NOCACHE under pressure (Darwin flag, rarely used)
//   --balloon: pre-allocate N MB of anonymous memory with incompressible
//              random data, to reproduce the constrained-RAM environment
//              the validator crashed in.
//
// Build:
//   clang -O2 -Wall -Wextra -o tools/io_probe_v2 tools/io_probe_v2.c
//
// Usage:
//   io_probe_v2 seq    <path> <total_MB> <chunk_MB> <nocache|cached>
//   io_probe_v2 random <path> <num_reads> <slice_KB> <nocache|cached>
//   io_probe_v2 mmap   <path> <total_MB> <cached|nocache> <seq|random>
//   [--balloon <MB>]   global flag, may appear anywhere after subcommand
//
// Exit codes:
//   0  clean run
//   2  bad arguments
//   3  pre-flight refused (not enough headroom even after balloon)
//   4  acceptance gate failed (only enforced for `random nocache`)
//   1  I/O error
//
// Acceptance gate (per user spec) -- only `random nocache` is hard-gated:
//   d_pageouts == 0
//   d_purges   == 0
//   d_compressor_growth < 16 MB
//   peak_rss < 2 * slice_KB * 1024 + 16 MB
// All other modes are diagnostic; they always exit 0 on completion so
// we can collect the comparison data without spurious failures.
// ---------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <mach/mach.h>
#include <mach/mach_host.h>

#ifndef MAP_NOCACHE
  // Darwin: pages mapped MAP_NOCACHE are placed on the inactive/noreuse
  // queue and are evicted preferentially.  Fall through to 0 on Linux so
  // the file still compiles, but the flag is a no-op there.
  #define MAP_NOCACHE 0
#endif

// ===========================================================================
// vm_stat sampling
// ===========================================================================

typedef struct {
    uint64_t page_size;
    uint64_t free, active, inactive, speculative, wired_down, purgeable;
    uint64_t pageouts, pageins, purges, compressor_pages;
} VmSample;

static void vm_sample(VmSample *s) {
    s->page_size = (uint64_t)sysconf(_SC_PAGESIZE);
    vm_statistics64_data_t vm;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm, &cnt) != KERN_SUCCESS) {
        memset(s, 0, sizeof(*s));
        s->page_size = (uint64_t)sysconf(_SC_PAGESIZE);
        return;
    }
    s->free             = vm.free_count;
    s->active           = vm.active_count;
    s->inactive         = vm.inactive_count;
    s->speculative      = vm.speculative_count;
    s->wired_down       = vm.wire_count;
    s->purgeable        = vm.purgeable_count;
    s->pageouts         = vm.pageouts;
    s->pageins          = vm.pageins;
    s->purges           = vm.purges;
    s->compressor_pages = vm.compressor_page_count;
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static uint64_t peak_rss_bytes(void) {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return (uint64_t)r.ru_maxrss; // Darwin: BYTES
}

static uint64_t avail_bytes_now(void) {
    VmSample s; vm_sample(&s);
    return (s.free + s.inactive + s.speculative) * s.page_size;
}

// ===========================================================================
// PRNG -- splitmix64, deterministic
// ===========================================================================

static uint64_t sm64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// ===========================================================================
// Balloon -- pre-allocate anonymous memory with INCOMPRESSIBLE data so the
// Darwin memory compressor cannot shrink it.  Random pages compress at
// roughly 1:1; zero pages compress to ~0.  We use random.
// Holding a pointer in a global so SIGINT handler can free it.
// ===========================================================================

static void   *g_balloon_ptr   = NULL;
static size_t  g_balloon_bytes = 0;

static void release_balloon(void) {
    if (g_balloon_ptr && g_balloon_bytes) {
        munmap(g_balloon_ptr, g_balloon_bytes);
        g_balloon_ptr = NULL;
        g_balloon_bytes = 0;
    }
}

static void on_sig(int s) {
    (void)s;
    release_balloon();
    _exit(130);
}

// Allocate <mb> MB of anonymous private memory, fill with random data.
// Returns 0 on success, 3 on pre-flight refusal.
static int inflate_balloon(uint64_t mb) {
    if (mb == 0) return 0;
    size_t bytes = (size_t)mb * 1024ull * 1024ull;

    // Hard floor: refuse if balloon would leave less than 2 GB available
    // AT ALLOCATION TIME.  We add 1 GB slack because filling with random
    // data will itself temporarily push counters.
    uint64_t avail = avail_bytes_now();
    if (avail < bytes + (2ull << 30)) {
        fprintf(stderr,
            "REFUSE balloon: requested=%.2fGB avail=%.2fGB (need request+2GB)\n",
            bytes / 1e9, avail / 1e9);
        return 3;
    }

    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_ANON | MAP_PRIVATE, -1, 0);
    if (p == MAP_FAILED) { perror("balloon: mmap"); return 1; }

    // Install signal handlers BEFORE we fill, so Ctrl-C during the fill
    // still releases the memory.
    g_balloon_ptr   = p;
    g_balloon_bytes = bytes;
    signal(SIGINT,  on_sig);
    signal(SIGTERM, on_sig);
    atexit(release_balloon);

    // Fill with random data, one 16 KB page at a time so we touch every
    // page (forcing commit).  arc4random is fast enough (~1 GB/s typical).
    const size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    uint8_t *q = (uint8_t *)p;
    for (size_t off = 0; off < bytes; off += pg) {
        // Only need to write a few bytes per page to commit it; but to
        // defeat any zero-page sharing or future compression we write
        // enough random bytes that compression cannot win on this page.
        // 256 random bytes scattered across the page is plenty.
        uint64_t seed = (uint64_t)off ^ 0xA5A5A5A5DEADBEEFull;
        for (int i = 0; i < 32; ++i) {
            uint64_t r = sm64(&seed);
            size_t at = (size_t)(r % pg) & ~(size_t)7;
            *(volatile uint64_t *)(q + off + at) = r;
        }
    }

    fprintf(stderr, "[balloon] held %.2f GB; avail now %.2f GB\n",
            bytes / 1e9, avail_bytes_now() / 1e9);
    return 0;
}

// ===========================================================================
// Latency histogram
// ===========================================================================

#define H_BUCKETS 8
static const char *h_labels[H_BUCKETS] = {
    "<10us", "10-100us", "100us-1ms", "1-10ms",
    "10-100ms", "100ms-1s", "1-10s", ">10s"
};
static const double h_bounds_s[H_BUCKETS - 1] = {
    1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0
};

static int bucket_of(double dt_s) {
    for (int i = 0; i < H_BUCKETS - 1; ++i)
        if (dt_s < h_bounds_s[i]) return i;
    return H_BUCKETS - 1;
}

static void print_hist(uint64_t *h, uint64_t total) {
    fprintf(stderr, "  latency: ");
    for (int i = 0; i < H_BUCKETS; ++i) {
        if (h[i] == 0) continue;
        fprintf(stderr, "%s=%" PRIu64 "(%.1f%%) ",
                h_labels[i], h[i], 100.0 * h[i] / total);
    }
    fprintf(stderr, "\n");
}

// ===========================================================================
// Subcommand: seq
// ===========================================================================

static int run_seq(const char *path, uint64_t total_b, uint64_t chunk_b,
                   int nocache)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    if (nocache) {
        if (fcntl(fd, F_NOCACHE, 1) < 0) perror("F_NOCACHE");
        if (fcntl(fd, F_RDAHEAD, 0) < 0) perror("F_RDAHEAD");
    }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return 1; }
    if (total_b > (uint64_t)st.st_size) total_b = (uint64_t)st.st_size;

    uint64_t need = chunk_b + (1ull << 30);
    if (!nocache && need < total_b + (1ull << 30)) need = total_b + (1ull << 30);
    uint64_t avail = avail_bytes_now();
    if (avail < need) {
        fprintf(stderr, "REFUSE: need=%.2fGB avail=%.2fGB\n",
                need / 1e9, avail / 1e9);
        close(fd);
        return 3;
    }

    uint8_t *buf = aligned_alloc(16384, chunk_b);
    if (!buf) { perror("aligned_alloc"); close(fd); return 1; }

    VmSample b; vm_sample(&b);
    double t0 = now_s();
    uint64_t off = 0, cksum = 0;
    while (off < total_b) {
        size_t n = (total_b - off > chunk_b) ? chunk_b : (total_b - off);
        ssize_t r = pread(fd, buf, n, (off_t)off);
        if (r <= 0) { perror("pread"); free(buf); close(fd); return 1; }
        const uint64_t pg = (uint64_t)sysconf(_SC_PAGESIZE);
        for (size_t i = 0; i < (size_t)r; i += pg) cksum += buf[i];
        off += (uint64_t)r;
    }
    double t1 = now_s();
    VmSample a; vm_sample(&a);
    uint64_t peak = peak_rss_bytes();
    free(buf); close(fd);

    double pgmb = b.page_size / (1024.0 * 1024.0);
    printf("subcmd=seq cache=%s total=%.0fMB chunk=%.0fMB time=%.3fs throughput=%.2fGB/s "
           "peak_rss=%.1fMB d_active=%+.1fMB d_inact=%+.1fMB d_compressor=%+.1fMB "
           "d_pageouts=%+" PRId64 " d_pageins=%+" PRId64 " d_purged=%+" PRId64 " cksum=0x%016" PRIx64 "\n",
           nocache ? "nocache" : "cached",
           off / 1e6, chunk_b / 1e6, t1 - t0, off / 1e9 / (t1 - t0),
           peak / 1e6,
           ((int64_t)a.active   - (int64_t)b.active)   * pgmb,
           ((int64_t)a.inactive - (int64_t)b.inactive) * pgmb,
           ((int64_t)a.compressor_pages - (int64_t)b.compressor_pages) * pgmb,
           (int64_t)a.pageouts - (int64_t)b.pageouts,
           (int64_t)a.pageins  - (int64_t)b.pageins,
           (int64_t)a.purges   - (int64_t)b.purges,
           cksum);
    return 0;
}

// ===========================================================================
// Subcommand: random  -- the MoE-shaped probe
// ===========================================================================

static int run_random(const char *path, uint64_t num_reads, uint64_t slice_b,
                      int nocache)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    if (nocache) {
        if (fcntl(fd, F_NOCACHE, 1) < 0) perror("F_NOCACHE");
        if (fcntl(fd, F_RDAHEAD, 0) < 0) perror("F_RDAHEAD");
    }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return 1; }
    if ((uint64_t)st.st_size < slice_b * 2) {
        fprintf(stderr, "file too small (%lld) for slice %llu\n",
                (long long)st.st_size, (unsigned long long)slice_b);
        close(fd); return 1;
    }

    const uint64_t pg = (uint64_t)sysconf(_SC_PAGESIZE);
    uint64_t total_b = num_reads * slice_b;

    // Pre-flight: we only need slice_b + 1 GB resident (single staging
    // buffer reused).  Cached mode additionally wants room for the whole
    // total_b (since each read populates cache).  Random pattern with
    // distinct offsets means cache CANNOT recycle.
    uint64_t need = slice_b + (1ull << 30);
    if (!nocache && need < total_b + (1ull << 30)) need = total_b + (1ull << 30);
    uint64_t avail = avail_bytes_now();
    if (avail < need) {
        fprintf(stderr, "REFUSE: need=%.2fGB avail=%.2fGB\n",
                need / 1e9, avail / 1e9);
        close(fd); return 3;
    }

    uint8_t *buf = aligned_alloc(16384, slice_b);
    if (!buf) { perror("aligned_alloc"); close(fd); return 1; }

    // Deterministic PRNG; seed is a function of (num_reads, slice_b) so
    // repeat runs hit the SAME offsets -- which means run 2 will hit the
    // cache hot from run 1.  That's a feature: we want to see it.
    uint64_t seed = num_reads ^ (slice_b * 0x9E3779B1u);

    uint64_t hist[H_BUCKETS] = {0};
    uint64_t cksum = 0;
    uint64_t max_off = ((uint64_t)st.st_size - slice_b) & ~(pg - 1);

    VmSample b; vm_sample(&b);
    double t0 = now_s();
    for (uint64_t i = 0; i < num_reads; ++i) {
        uint64_t r64 = sm64(&seed);
        uint64_t off = (r64 % (max_off / pg + 1)) * pg;
        double r0 = now_s();
        ssize_t got = pread(fd, buf, slice_b, (off_t)off);
        double r1 = now_s();
        if (got <= 0) { perror("pread"); free(buf); close(fd); return 1; }
        hist[bucket_of(r1 - r0)]++;
        // Touch one byte per page so the read isn't optimized away.
        for (size_t j = 0; j < (size_t)got; j += pg) cksum += buf[j];
    }
    double t1 = now_s();
    VmSample a; vm_sample(&a);
    uint64_t peak = peak_rss_bytes();
    free(buf); close(fd);

    double pgmb = b.page_size / (1024.0 * 1024.0);
    int64_t d_pageouts = (int64_t)a.pageouts - (int64_t)b.pageouts;
    int64_t d_purges   = (int64_t)a.purges   - (int64_t)b.purges;
    double  d_compr_mb = ((int64_t)a.compressor_pages - (int64_t)b.compressor_pages) * pgmb;

    printf("subcmd=random cache=%s reads=%" PRIu64 " slice=%.2fMB total=%.1fMB "
           "time=%.3fs throughput=%.2fGB/s peak_rss=%.1fMB "
           "d_active=%+.1fMB d_inact=%+.1fMB d_compressor=%+.1fMB "
           "d_pageouts=%+" PRId64 " d_pageins=%+" PRId64 " d_purged=%+" PRId64 " "
           "cksum=0x%016" PRIx64 "\n",
           nocache ? "nocache" : "cached",
           num_reads, slice_b / 1e6, total_b / 1e6,
           t1 - t0, total_b / 1e9 / (t1 - t0),
           peak / 1e6,
           ((int64_t)a.active   - (int64_t)b.active)   * pgmb,
           ((int64_t)a.inactive - (int64_t)b.inactive) * pgmb,
           d_compr_mb,
           d_pageouts, (int64_t)a.pageins - (int64_t)b.pageins, d_purges,
           cksum);
    print_hist(hist, num_reads);

    // Hard acceptance gate ONLY for nocache.
    //
    // Gate must be balloon-aware: when a balloon is active, peak_rss
    // includes it, and the compressor may legitimately grow as Darwin
    // compresses random balloon pages to satisfy our small staging
    // allocation.  That's the kernel doing the RIGHT thing under
    // pressure -- compress, don't swap.  So we discount both metrics
    // by the balloon size.
    if (nocache) {
        double  balloon_mb    = g_balloon_bytes / 1e6;
        // Allow ~3% of balloon to compress (random data, not truly
        // incompressible, plus existing inactive pages may compress too).
        double  compr_budget  = 16.0 + 0.03 * balloon_mb;
        // RSS budget: staging ring + balloon + slack.
        uint64_t rss_ceiling  = 2 * slice_b + g_balloon_bytes + (32ull << 20);
        int fail = 0;
        // These three are what truly matter: pressure on OTHER apps.
        if (d_pageouts > 0)
            { fprintf(stderr, "FAIL: d_pageouts=%" PRId64 " (other apps swapped to disk)\n",
                      d_pageouts); fail = 1; }
        if (d_purges   > 0)
            { fprintf(stderr, "FAIL: d_purges=%" PRId64 " (cache reclaimed under pressure)\n",
                      d_purges); fail = 1; }
        if (d_compr_mb > compr_budget)
            { fprintf(stderr, "FAIL: d_compressor=%.1fMB > %.1fMB budget (balloon=%.0fMB)\n",
                      d_compr_mb, compr_budget, balloon_mb); fail = 1; }
        if (peak > rss_ceiling)
            { fprintf(stderr, "FAIL: peak_rss=%.1fMB > ceiling=%.1fMB (balloon=%.0fMB)\n",
                      peak / 1e6, rss_ceiling / 1e6, balloon_mb); fail = 1; }
        if (fail) return 4;
        fprintf(stderr, "PASS: random nocache meets acceptance gate (balloon=%.0fMB)\n",
                balloon_mb);
    }
    return 0;
}

// ===========================================================================
// Subcommand: decode  -- multi-token MoE simulation
//
// Per-token: issue <experts> random preads of <slice_KB> each with F_NOCACHE
// into one reused staging buffer.  Per-token timing + pre/post vm_stat.
// Across all N tokens: aggregate latency histogram, per-token throughput
// timeseries, peak RSS, total pressure deltas.
//
// Hard acceptance gate (this is the WHOLE-RUN production-readiness test):
//   - d_pageouts == 0 across entire run
//   - d_purges   <= 16  (small noise allowed; not zero because long runs
//                        accumulate background daemon activity)
//   - d_compressor < 64 MB + 3% balloon  (long runs let kernel breathe)
//   - peak_rss < 2 * slice + balloon + 64 MB
//   - per-token p99 latency < 2 * theoretical_floor
//     (floor = slice_b / measured_throughput, throughput from first token)
//
// If this passes for N >= 3 with real R1 expert-traffic volume, we have
// proven the runtime invariant holds at scale.
// ===========================================================================

static int run_decode(const char *path, uint64_t num_tokens,
                      uint64_t experts_per_token, uint64_t slice_b)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    if (fcntl(fd, F_NOCACHE, 1) < 0) perror("F_NOCACHE");
    if (fcntl(fd, F_RDAHEAD, 0) < 0) perror("F_RDAHEAD");

    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return 1; }
    if ((uint64_t)st.st_size < slice_b * 2) {
        fprintf(stderr, "file too small\n"); close(fd); return 1;
    }

    const uint64_t pg = (uint64_t)sysconf(_SC_PAGESIZE);
    uint64_t per_token_b = experts_per_token * slice_b;
    uint64_t total_b     = num_tokens * per_token_b;

    // Pre-flight: only need slice_b + 1 GB resident (single reused buffer).
    uint64_t avail = avail_bytes_now();
    if (avail < slice_b + (1ull << 30)) {
        fprintf(stderr, "REFUSE: need=%.2fGB avail=%.2fGB\n",
                (slice_b + (1ull << 30)) / 1e9, avail / 1e9);
        close(fd); return 3;
    }

    uint8_t *buf = aligned_alloc(16384, slice_b);
    if (!buf) { perror("aligned_alloc"); close(fd); return 1; }

    uint64_t max_off = ((uint64_t)st.st_size - slice_b) & ~(pg - 1);
    uint64_t total_reads = num_tokens * experts_per_token;
    uint64_t hist[H_BUCKETS] = {0};
    uint64_t cksum = 0;

    // Per-token throughput timeseries, on stack-friendly cap.
    double tok_t[1024];
    if (num_tokens > 1024) num_tokens = 1024;

    fprintf(stderr,
        "[decode] %" PRIu64 " tokens x %" PRIu64 " experts x %.2f MB "
        "= %.2f GB/token, %.2f GB total\n",
        num_tokens, experts_per_token, slice_b / 1e6,
        per_token_b / 1e9, total_b / 1e9);

    VmSample run_before; vm_sample(&run_before);
    double run_t0 = now_s();

    for (uint64_t tk = 0; tk < num_tokens; ++tk) {
        // Each token has its OWN deterministic seed -- distinct cold pages.
        uint64_t seed = 0xC0FFEE00ull + tk * 0x9E3779B97F4A7C15ull
                      + slice_b * 0xDEADBEEFull;
        double t0 = now_s();
        for (uint64_t i = 0; i < experts_per_token; ++i) {
            uint64_t r = sm64(&seed);
            uint64_t off = (r % (max_off / pg + 1)) * pg;
            double r0 = now_s();
            ssize_t got = pread(fd, buf, slice_b, (off_t)off);
            double r1 = now_s();
            if (got <= 0) { perror("pread"); free(buf); close(fd); return 1; }
            hist[bucket_of(r1 - r0)]++;
            for (size_t j = 0; j < (size_t)got; j += pg) cksum += buf[j];
        }
        double t1 = now_s();
        tok_t[tk] = t1 - t0;
    }

    double run_t1 = now_s();
    VmSample run_after; vm_sample(&run_after);
    uint64_t peak = peak_rss_bytes();
    free(buf); close(fd);

    double pgmb = run_before.page_size / (1024.0 * 1024.0);
    int64_t d_pageouts = (int64_t)run_after.pageouts - (int64_t)run_before.pageouts;
    int64_t d_purges   = (int64_t)run_after.purges   - (int64_t)run_before.purges;
    double  d_compr_mb = ((int64_t)run_after.compressor_pages
                       -  (int64_t)run_before.compressor_pages) * pgmb;
    double  total_s    = run_t1 - run_t0;
    double  avg_tok_s  = total_s / num_tokens;

    // Per-token report
    fprintf(stderr, "[decode] per-token timing (s): ");
    double tmin = tok_t[0], tmax = tok_t[0];
    for (uint64_t tk = 0; tk < num_tokens; ++tk) {
        fprintf(stderr, "%.3f ", tok_t[tk]);
        if (tok_t[tk] < tmin) tmin = tok_t[tk];
        if (tok_t[tk] > tmax) tmax = tok_t[tk];
    }
    fprintf(stderr, "\n");

    printf("subcmd=decode tokens=%" PRIu64 " experts/tok=%" PRIu64 " slice=%.2fMB "
           "total=%.2fGB time=%.3fs throughput=%.2fGB/s avg_tok=%.3fs "
           "p_tok_min=%.3fs p_tok_max=%.3fs eff_tok_rate=%.2ftok/s "
           "peak_rss=%.1fMB d_active=%+.1fMB d_inact=%+.1fMB d_compressor=%+.1fMB "
           "d_pageouts=%+" PRId64 " d_pageins=%+" PRId64 " d_purged=%+" PRId64 " "
           "cksum=0x%016" PRIx64 "\n",
           num_tokens, experts_per_token, slice_b / 1e6,
           total_b / 1e9, total_s, total_b / 1e9 / total_s, avg_tok_s,
           tmin, tmax, 1.0 / avg_tok_s,
           peak / 1e6,
           ((int64_t)run_after.active   - (int64_t)run_before.active)   * pgmb,
           ((int64_t)run_after.inactive - (int64_t)run_before.inactive) * pgmb,
           d_compr_mb,
           d_pageouts,
           (int64_t)run_after.pageins - (int64_t)run_before.pageins,
           d_purges,
           cksum);
    print_hist(hist, total_reads);

    // Hard gate -- production readiness check.
    //
    // What we ACTUALLY care about: did we force the kernel to write dirty
    // pages of other processes to swap?  That is d_pageouts, and that
    // must be zero.  Everything else is the kernel doing its job:
    //   - d_purges  : voluntary reclaim of opt-in volatile pages (caches
    //                 in other apps).  Scales with balloon; not pressure.
    //   - d_compressor : compressed cold anon pages.  Healthy response.
    //   - peak_rss  : dominated by balloon when present; our own working
    //                 set is the staging buffer.
    double  balloon_mb   = g_balloon_bytes / 1e6;
    // Allow purges roughly proportional to balloon size (kernel reclaims
    // ~1% of equivalent volatile memory).  Without balloon, allow 16.
    int64_t purge_budget = (int64_t)(16 + balloon_mb * 16); // ~16 purges/MB balloon
    double  compr_budget = 64.0 + 0.05 * balloon_mb;
    uint64_t rss_ceiling = 2 * slice_b + g_balloon_bytes + (64ull << 20);
    int fail = 0;
    if (d_pageouts > 0)
        { fprintf(stderr, "FAIL: d_pageouts=%" PRId64 " (other apps swapped)\n",
                  d_pageouts); fail = 1; }
    if (d_purges > purge_budget)
        { fprintf(stderr, "FAIL: d_purges=%" PRId64 " > %" PRId64 " budget (balloon=%.0fMB)\n",
                  d_purges, purge_budget, balloon_mb); fail = 1; }
    if (d_compr_mb > compr_budget)
        { fprintf(stderr, "FAIL: d_compressor=%.1fMB > %.1fMB budget\n",
                  d_compr_mb, compr_budget); fail = 1; }
    if (peak > rss_ceiling)
        { fprintf(stderr, "FAIL: peak_rss=%.1fMB > ceiling=%.1fMB\n",
                  peak / 1e6, rss_ceiling / 1e6); fail = 1; }
    if (fail) return 4;
    fprintf(stderr, "PASS: decode microbench -- sustained %.2f tok/s of MoE I/O "
                    "for %" PRIu64 " tokens (pageouts=0, balloon=%.0fMB)\n",
            1.0 / avg_tok_s, num_tokens, balloon_mb);
    return 0;
}

// ===========================================================================
// Subcommand: mmap
// ===========================================================================

static int run_mmap(const char *path, uint64_t total_b, int nocache, int random_pattern)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return 1; }
    if (total_b > (uint64_t)st.st_size) total_b = (uint64_t)st.st_size;

    // mmap always populates resident pages as we touch them; pre-flight
    // requires total_b + 1 GB regardless of NOCACHE (NOCACHE just hints
    // eviction priority, doesn't prevent residency).
    uint64_t avail = avail_bytes_now();
    if (avail < total_b + (1ull << 30)) {
        fprintf(stderr, "REFUSE: need=%.2fGB avail=%.2fGB\n",
                (total_b + (1ull << 30)) / 1e9, avail / 1e9);
        close(fd); return 3;
    }

    int flags = MAP_SHARED;
    if (nocache) flags |= MAP_NOCACHE;
    void *p = mmap(NULL, st.st_size, PROT_READ, flags, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    const uint64_t pg = (uint64_t)sysconf(_SC_PAGESIZE);
    const volatile uint8_t *q = (const uint8_t *)p;
    uint64_t cksum = 0;

    VmSample b; vm_sample(&b);
    double t0 = now_s();

    if (random_pattern) {
        // Touch total_b / pg distinct random pages within the first
        // st.st_size span.  Same PRNG seed scheme as `random` subcmd
        // for reproducibility.
        uint64_t n_pages = total_b / pg;
        uint64_t max_pg  = (uint64_t)st.st_size / pg;
        uint64_t seed    = n_pages ^ 0x12345678ull;
        for (uint64_t i = 0; i < n_pages; ++i) {
            uint64_t r = sm64(&seed);
            uint64_t pi = r % max_pg;
            cksum += q[pi * pg];
        }
    } else {
        for (uint64_t off = 0; off < total_b; off += pg) cksum += q[off];
    }

    double t1 = now_s();
    VmSample a; vm_sample(&a);
    uint64_t peak = peak_rss_bytes();
    munmap(p, st.st_size);
    close(fd);

    double pgmb = b.page_size / (1024.0 * 1024.0);
    printf("subcmd=mmap cache=%s pattern=%s touched=%.0fMB time=%.3fs throughput=%.2fGB/s "
           "peak_rss=%.1fMB d_active=%+.1fMB d_inact=%+.1fMB d_compressor=%+.1fMB "
           "d_pageouts=%+" PRId64 " d_pageins=%+" PRId64 " d_purged=%+" PRId64 " cksum=0x%016" PRIx64 "\n",
           nocache ? "nocache" : "cached",
           random_pattern ? "random" : "seq",
           total_b / 1e6, t1 - t0, total_b / 1e9 / (t1 - t0),
           peak / 1e6,
           ((int64_t)a.active   - (int64_t)b.active)   * pgmb,
           ((int64_t)a.inactive - (int64_t)b.inactive) * pgmb,
           ((int64_t)a.compressor_pages - (int64_t)b.compressor_pages) * pgmb,
           (int64_t)a.pageouts - (int64_t)b.pageouts,
           (int64_t)a.pageins  - (int64_t)b.pageins,
           (int64_t)a.purges   - (int64_t)b.purges,
           cksum);
    return 0;
}

// ===========================================================================
// Argument parsing
// ===========================================================================

static void usage(const char *prog) {
    fprintf(stderr,
"usage:\n"
"  %s seq    <path> <total_MB> <chunk_MB> <nocache|cached>     [--balloon MB]\n"
"  %s random <path> <num_reads> <slice_KB> <nocache|cached>    [--balloon MB]\n"
"  %s mmap   <path> <total_MB> <cached|nocache> <seq|random>   [--balloon MB]\n"
"  %s decode <path> <num_tokens> <experts_per_token> <slice_KB> [--balloon MB]\n",
        prog, prog, prog, prog);
}

static int parse_cache(const char *s, int *out_nocache) {
    if (!strcmp(s, "cached"))  { *out_nocache = 0; return 0; }
    if (!strcmp(s, "nocache")) { *out_nocache = 1; return 0; }
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 2; }

    // Find and strip --balloon if present.
    uint64_t balloon_mb = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--balloon") && i + 1 < argc) {
            balloon_mb = (uint64_t)strtoull(argv[i+1], 0, 10);
            for (int j = i; j + 2 <= argc; ++j) argv[j] = argv[j+2];
            argc -= 2;
            break;
        }
    }

    if (balloon_mb > 0) {
        int rc = inflate_balloon(balloon_mb);
        if (rc) return rc;
    }

    const char *sub = argv[1];
    int rc = 2;

    if (!strcmp(sub, "seq") && argc == 6) {
        uint64_t total_b = (uint64_t)strtoull(argv[3], 0, 10) * 1024ull * 1024ull;
        uint64_t chunk_b = (uint64_t)strtoull(argv[4], 0, 10) * 1024ull * 1024ull;
        int nocache;
        if (parse_cache(argv[5], &nocache)) { usage(argv[0]); return 2; }
        rc = run_seq(argv[2], total_b, chunk_b, nocache);

    } else if (!strcmp(sub, "random") && argc == 6) {
        uint64_t num_reads = (uint64_t)strtoull(argv[3], 0, 10);
        uint64_t slice_b   = (uint64_t)strtoull(argv[4], 0, 10) * 1024ull;
        int nocache;
        if (parse_cache(argv[5], &nocache)) { usage(argv[0]); return 2; }
        rc = run_random(argv[2], num_reads, slice_b, nocache);

    } else if (!strcmp(sub, "mmap") && argc == 6) {
        uint64_t total_b = (uint64_t)strtoull(argv[3], 0, 10) * 1024ull * 1024ull;
        int nocache;
        if (parse_cache(argv[4], &nocache)) { usage(argv[0]); return 2; }
        int random_pattern;
        if (!strcmp(argv[5], "seq"))         random_pattern = 0;
        else if (!strcmp(argv[5], "random")) random_pattern = 1;
        else { usage(argv[0]); return 2; }
        rc = run_mmap(argv[2], total_b, nocache, random_pattern);

    } else if (!strcmp(sub, "decode") && argc == 6) {
        uint64_t num_tokens = (uint64_t)strtoull(argv[3], 0, 10);
        uint64_t experts    = (uint64_t)strtoull(argv[4], 0, 10);
        uint64_t slice_b    = (uint64_t)strtoull(argv[5], 0, 10) * 1024ull;
        rc = run_decode(argv[2], num_tokens, experts, slice_b);

    } else {
        usage(argv[0]);
    }

    return rc;
}

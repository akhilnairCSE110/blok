// Always-on profiling. Designed so a freeze leaves a complete trail in stderr:
// every log line is flushed immediately so the last line is the last thing
// the process did before locking up.  Zero env flags, zero opt-in.  Cost is
// one fprintf per phase boundary -- microseconds, dwarfed by anything we'd
// want to measure.
#pragma once
#include <chrono>
#include <cstdio>
#include <mach/mach.h>

namespace blade::prof {

inline long long now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

// Resident set in MB via mach task_info.  ~1us per call; safe to sample often.
inline size_t rss_mb() {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &cnt) != KERN_SUCCESS) return 0;
    return (size_t)(info.resident_size >> 20);
}

// Always-flush log.  Keep diagnostics distinct from generated model text.
template<class... A>
inline void log(const char* fmt, A... args) {
    std::fprintf(stderr, "[metalblok] ");
    std::fprintf(stderr, fmt, args...);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

// Pre-action breadcrumb so a hang is bracketed exactly between a "begin" and
// the missing "end" line.  Use for any operation that could take >100ms.
inline void mark(const char* what) {
    std::fprintf(stderr, "[metalblok] >>> %s  (rss=%zuMB)\n", what, rss_mb());
    std::fflush(stderr);
}

} // namespace blade::prof

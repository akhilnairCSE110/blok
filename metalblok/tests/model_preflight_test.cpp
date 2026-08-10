#include "preflight.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <string>

int main() {
    char path[] = "/tmp/metalblok-preflight-XXXXXX";
    int fd = ::mkstemp(path);
    if (fd < 0) return 1;
    if (::ftruncate(fd, 32LL << 20) != 0) return 2;
    ::close(fd);

    auto sparse = blade::inspect_model_files(path);
    if (sparse.shards.size() != 1 || sparse.all_resident ||
        !sparse.shards[0].sparse) return 3;

    ::unlink(path);
    auto missing = blade::inspect_model_files(path);
    if (missing.shards.empty() || missing.shards[0].exists) return 4;
    return 0;
}

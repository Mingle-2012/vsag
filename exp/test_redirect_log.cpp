#include <iostream>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_sinks.h>

int main() {
    int fd = ::open("app.log", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) {
        perror("open");
        return 1;
    }

    ::dup2(fd, STDOUT_FILENO);
    ::dup2(fd, STDERR_FILENO);
    ::close(fd);

    auto sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("log", sink);
    spdlog::set_default_logger(logger);

    std::cout << "Hello from cout" << std::endl;
    std::cerr << "Hello from cerr" << std::endl;
    spdlog::info("Hello from spdlog!");

    return 0;
}

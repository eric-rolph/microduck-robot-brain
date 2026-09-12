#pragma once

#include "duck_shm.hpp"
#include <stdexcept>
#include <string>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

class ShmClient {
public:
    explicit ShmClient(bool is_master) : is_master_(is_master) {
        int flags = O_RDWR;
        if (is_master_) {
            flags |= O_CREAT;
        }

        fd_ = shm_open(MICRODUCK_SHM_NAME, flags, 0666);
        if (fd_ < 0) {
            throw std::runtime_error("shm_open failed for Microduck IPC");
        }

        if (is_master_) {
            if (ftruncate(fd_, sizeof(DuckSharedMemory)) != 0) {
                close(fd_);
                throw std::runtime_error("ftruncate failed");
            }
        }

        void* ptr = mmap(nullptr, sizeof(DuckSharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (ptr == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("mmap failed");
        }

        shm_ = static_cast<DuckSharedMemory*>(ptr);

        if (is_master_) {
            new (shm_) DuckSharedMemory(); // Initialize atomic variables in mapped memory
        }
    }

    ~ShmClient() {
        if (shm_ != nullptr) {
            munmap(shm_, sizeof(DuckSharedMemory));
            shm_ = nullptr;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
        if (is_master_) {
            shm_unlink(MICRODUCK_SHM_NAME);
        }
    }

    DuckSharedMemory* get() { return shm_; }

private:
    int fd_{-1};
    bool is_master_{false};
    DuckSharedMemory* shm_{nullptr};
};

#else
// Windows fallback implementation using Windows Named Shared Memory
#include <windows.h>

class ShmClient {
public:
    explicit ShmClient(bool is_master) : is_master_(is_master) {
        const char* map_name = "Local\\microduck_ipc_shm";
        if (is_master_) {
            h_map_ = CreateFileMappingA(
                INVALID_HANDLE_VALUE,
                nullptr,
                PAGE_READWRITE,
                0,
                sizeof(DuckSharedMemory),
                map_name
            );
            if (h_map_ == nullptr) {
                throw std::runtime_error("CreateFileMapping failed on Windows");
            }
        } else {
            h_map_ = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, map_name);
            if (h_map_ == nullptr) {
                throw std::runtime_error("OpenFileMapping failed on Windows");
            }
        }

        void* ptr = MapViewOfFile(h_map_, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(DuckSharedMemory));
        if (ptr == nullptr) {
            CloseHandle(h_map_);
            throw std::runtime_error("MapViewOfFile failed on Windows");
        }

        shm_ = static_cast<DuckSharedMemory*>(ptr);
        if (is_master_) {
            new (shm_) DuckSharedMemory();
        }
    }

    ~ShmClient() {
        if (shm_ != nullptr) {
            UnmapViewOfFile(shm_);
            shm_ = nullptr;
        }
        if (h_map_ != nullptr) {
            CloseHandle(h_map_);
            h_map_ = nullptr;
        }
    }

    DuckSharedMemory* get() { return shm_; }

private:
    HANDLE h_map_{nullptr};
    bool is_master_{false};
    DuckSharedMemory* shm_{nullptr};
};
#endif

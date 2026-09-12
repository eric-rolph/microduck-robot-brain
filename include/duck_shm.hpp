#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>

#define MICRODUCK_SHM_NAME "/microduck_ipc_shm"

// Command payload: BehaviorTree -> robotd
struct alignas(64) MotionCommand {
    float vx{0.0f};            // Linear x velocity (m/s)
    float vy{0.0f};            // Linear y velocity (m/s)
    float wz{0.0f};            // Yaw angular velocity (rad/s)
    float roll_cmd{0.0f};      // Lean trim roll (rad)
    float pitch_cmd{0.0f};     // Lean trim pitch (rad)
    bool emergency_stop{false};
    uint64_t timestamp_ns{0};  // Watchdog timestamp
};

// State payload: robotd -> BehaviorTree
struct alignas(64) RobotFeedback {
    float projected_gravity[3]{0.0f, 0.0f, -1.0f};
    float base_lin_vel[3]{0.0f};
    float base_ang_vel[3]{0.0f};
    float joint_pos[12]{0.0f};
    float joint_vel[12]{0.0f};
    uint8_t stability_state{0}; // 0 = HIGH, 1 = MEDIUM, 2 = LOW
    uint8_t roughness_state{0}; // 0 = LOW, 1 = MEDIUM, 2 = HIGH
    uint64_t timestamp_ns{0};
};

// Lock-Free SeqLock Channel (Non-blocking, single-writer / multi-reader)
template <typename T>
struct alignas(64) SeqLockChannel {
    std::atomic<uint32_t> sequence{0};
    T data;

    void write(const T& payload) {
        uint32_t seq = sequence.load(std::memory_order_relaxed);
        sequence.store(seq + 1, std::memory_order_release); // Odd: write in progress
        std::atomic_thread_fence(std::memory_order_release);

        data = payload;

        std::atomic_thread_fence(std::memory_order_release);
        sequence.store(seq + 2, std::memory_order_release); // Even: write complete
    }

    bool read(T& dest) const {
        uint32_t s1 = 0;
        uint32_t s2 = 0;
        constexpr int MAX_RETRIES = 10;
        int retries = 0;

        do {
            s1 = sequence.load(std::memory_order_acquire);
            if (s1 & 1) { // Writer currently updating
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            dest = data;
            std::atomic_thread_fence(std::memory_order_acquire);
            s2 = sequence.load(std::memory_order_acquire);
            retries++;
        } while ((s1 != s2 || (s1 & 1)) && retries < MAX_RETRIES);

        return (s1 == s2 && !(s1 & 1));
    }
};

struct alignas(64) DuckSharedMemory {
    SeqLockChannel<MotionCommand> cmd_channel;       // Written by BT, read by robotd
    SeqLockChannel<RobotFeedback> feedback_channel;  // Written by robotd, read by BT
};

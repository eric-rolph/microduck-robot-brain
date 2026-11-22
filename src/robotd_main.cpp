// robotd_main.cpp - SeqLock shared-memory RT loop harness (50 Hz IPC latency benchmark/stub).
// Production physical Microduck deployment communicates with Pollen Robotics' robotd over
// /run/robotd.sock via microduck_brain/pollen_bridge.py.
#include "shm_client.hpp"
#include <chrono>
#include <iostream>
#include <thread>
#include <atomic>
#include <csignal>

#ifndef _WIN32
#include <sys/mman.h>
#endif

namespace {
std::atomic<bool> g_running{true};

void handle_signal(int) {
    g_running.store(false);
}

uint64_t get_time_ns() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}
} // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

#ifndef _WIN32
    // Lock process pages into RAM to eliminate page fault jitter
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        std::cerr << "Warning: Could not lock memory via mlockall (requires CAP_IPC_LOCK or root)\n";
    }
#endif

    // Initialize master shared memory segment
    ShmClient shm(true);
    DuckSharedMemory* mem = shm.get();

    MotionCommand cmd{};
    MotionCommand last_valid_cmd{};
    bool has_valid_cmd = false;
    RobotFeedback feedback{};

    std::cout << "[robotd] Real-time engine started. Initialized shared memory.\n";

    constexpr uint64_t LOOP_PERIOD_NS = 20'000'000;    // 50 Hz (20 ms)
    constexpr uint64_t COMMAND_TIMEOUT_NS = 250'000'000; // 250 ms watchdog

    uint64_t loop_count = 0;

    while (g_running.load()) {
        auto start_time = get_time_ns();

        // 1. Non-blocking read from BT command channel
        MotionCommand latest_cmd{};
        if (mem->cmd_channel.read(latest_cmd)) {
            last_valid_cmd = latest_cmd;
            has_valid_cmd = true;
        }

        cmd = has_valid_cmd ? last_valid_cmd : MotionCommand{};
        const bool timed_out = !has_valid_cmd
            || cmd.timestamp_ns == 0
            || (start_time > cmd.timestamp_ns && (start_time - cmd.timestamp_ns > COMMAND_TIMEOUT_NS));

        if (timed_out || cmd.emergency_stop) {
            cmd.vx = 0.0f;
            cmd.vy = 0.0f;
            cmd.wz = 0.0f;
            cmd.roll_cmd = 0.0f;
            cmd.pitch_cmd = 0.0f;
        }

        // 2. Control evaluation: In production, feeds observation vector to ONNX engine
        // Mock proprioceptive state updates
        feedback.projected_gravity[0] = 0.0f;
        feedback.projected_gravity[1] = 0.0f;
        feedback.projected_gravity[2] = -1.0f;
        feedback.base_lin_vel[0] = cmd.vx;
        feedback.base_lin_vel[1] = cmd.vy;
        feedback.base_lin_vel[2] = 0.0f;
        feedback.base_ang_vel[2] = cmd.wz;

        // Stability state: 0 = HIGH, 1 = MEDIUM, 2 = LOW
        feedback.stability_state = 0;
        feedback.roughness_state = 0;
        feedback.timestamp_ns = get_time_ns();

        // 3. Write feedback to BT
        mem->feedback_channel.write(feedback);

        loop_count++;
        if (loop_count % 250 == 0) { // Print status every 5 seconds at 50 Hz
            std::cout << "[robotd] Heartbeat: loop " << loop_count
                      << " | cmd vx: " << cmd.vx
                      << " wz: " << cmd.wz
                      << " roll: " << cmd.roll_cmd << "\n";
        }

        // 4. Precise cycle sleep
        auto elapsed = get_time_ns() - start_time;
        if (elapsed < LOOP_PERIOD_NS) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(LOOP_PERIOD_NS - elapsed));
        }
    }

    std::cout << "[robotd] Exiting cleanly.\n";
    return 0;
}

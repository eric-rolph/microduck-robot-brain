// bt_bridge_node.cpp - Microduck BehaviorTree shared memory bridge node (10-20 Hz)
#include "shm_client.hpp"
#include <chrono>
#include <iostream>
#include <thread>
#include <atomic>
#include <csignal>

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

class WalkActionBTNode {
public:
    explicit WalkActionBTNode(DuckSharedMemory* mem) : mem_(mem) {}

    bool tick(float target_vx, float target_wz, float head_roll) {
        RobotFeedback feedback;
        if (mem_->feedback_channel.read(feedback)) {
            // Stability check: if balance is compromised (2 = LOW), clamp lean to zero
            if (feedback.stability_state == 2) {
                head_roll = 0.0f;
            }
        }

        MotionCommand cmd;
        cmd.vx = target_vx;
        cmd.vy = 0.0f;
        cmd.wz = target_wz;
        cmd.roll_cmd = head_roll;
        cmd.pitch_cmd = 0.0f;
        cmd.emergency_stop = false;
        cmd.timestamp_ns = get_time_ns();

        mem_->cmd_channel.write(cmd);
        return true;
    }

private:
    DuckSharedMemory* mem_{nullptr};
};

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    std::cout << "[BT Bridge] Connecting to shared memory...\n";
    ShmClient shm(false);
    DuckSharedMemory* mem = shm.get();

    WalkActionBTNode walk_node(mem);
    std::cout << "[BT Bridge] Connected. Running 20 Hz tick loop.\n";

    constexpr uint64_t LOOP_PERIOD_NS = 50'000'000; // 20 Hz (50 ms)
    uint64_t tick_count = 0;

    while (g_running.load() && tick_count < 200) { // Default to 200 ticks in demo mode
        auto start_time = get_time_ns();

        // Send 0.4 m/s forward walk with 0.10 rad ambient roll trim
        walk_node.tick(0.4f, 0.0f, 0.10f);
        tick_count++;

        if (tick_count % 20 == 0) {
            std::cout << "[BT Bridge] Tick " << tick_count << " dispatched\n";
        }

        auto elapsed = get_time_ns() - start_time;
        if (elapsed < LOOP_PERIOD_NS) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(LOOP_PERIOD_NS - elapsed));
        }
    }

    std::cout << "[BT Bridge] Completed ticks or received stop signal.\n";
    return 0;
}

"""Black-box Actuator Model (BAM M6) for Dynamixel XL330 servos.

Simulates DC motor electrical characteristics, battery voltage sag under load,
back-EMF velocity-dependent torque limits, gearbox friction, and transport lag.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class BamM6Config:
    """Parameters for Dynamixel XL330 servo and battery circuit."""

    # Battery circuit
    v_open_circuit: float = 8.2      # Fully charged 2S LiPo voltage (V)
    v_min: float = 6.0               # Brownout cutoff voltage (V)
    r_internal_batt: float = 0.18    # Battery internal resistance (Ohms)
    idle_current: float = 0.15       # Standby electronics current draw (A)

    # Motor characteristics (XL330-M288)
    v_nominal: float = 7.4           # Rated voltage (V)
    stall_torque: float = 0.52       # Stall torque at 7.4V (N*m)
    max_stall_current: float = 1.4   # Maximum physical stall current per servo (A)
    no_load_speed: float = 6.39      # Max angular velocity at 7.4V (rad/s)
    motor_resistance: float = 5.69   # Motor winding resistance (Ohms)
    torque_constant: float = 0.40    # Kt in N*m / A
    back_emf_constant: float = 1.15  # Ke in V / (rad/s)

    # Firmware PD control (output-side position tracking)
    kp: float = 25.0                 # Position proportional gain (N*m / rad)
    kd: float = 0.6                  # Velocity derivative gain (N*m*s / rad)

    # Gearbox friction and stiction
    coulomb_friction: float = 0.035  # Dynamic Coulomb friction (N*m)
    stiction_torque: float = 0.055   # Breakaway static friction (N*m)
    viscous_damping: float = 0.006   # Viscous drag coefficient (N*m*s / rad)
    stiction_vel_threshold: float = 0.01  # Velocity threshold for stiction (rad/s)

    # Actuator lag
    delay_steps: int = 1             # Discrete step delay (at 50 Hz, 1 step = 20 ms)


class BamM6ActuatorModel:
    """Simulates BAM M6 actuator dynamics for 14 Dynamixel XL330 servos."""

    def __init__(self, num_actuators: int = 14, config: BamM6Config | None = None) -> None:
        self.num_actuators = num_actuators
        self.cfg = config or BamM6Config()

        # State tracking
        self.battery_voltage: float = self.cfg.v_open_circuit
        self.last_torques: np.ndarray = np.zeros(num_actuators, dtype=np.float32)
        self.last_torque_limits: np.ndarray = np.full(num_actuators, self.cfg.stall_torque, dtype=np.float32)
        self.last_current: float = self.cfg.idle_current

        # Delay queue for target setpoints
        self._delay_queue: deque[np.ndarray] = deque()
        self.reset()

    def reset(self, initial_targets: np.ndarray | None = None) -> None:
        """Reset internal battery state and delay queue."""
        self.battery_voltage = self.cfg.v_open_circuit
        self.last_torques = np.zeros(self.num_actuators, dtype=np.float32)
        self.last_torque_limits = np.full(self.num_actuators, self.cfg.stall_torque, dtype=np.float32)
        self.last_current = self.cfg.idle_current

        self._delay_queue.clear()
        base = initial_targets if initial_targets is not None else np.zeros(self.num_actuators, dtype=np.float32)
        for _ in range(self.cfg.delay_steps):
            self._delay_queue.append(base.copy())

    def step_delay(self, target_positions: np.ndarray) -> np.ndarray:
        """Advance transport delay queue once per policy step (50 Hz)."""
        self._delay_queue.append(target_positions.copy())
        return self._delay_queue.popleft()

    def update_battery_voltage(self, active_torques: np.ndarray) -> float:
        """Compute battery terminal voltage after load-dependent internal resistance drop."""
        # Total motor current: bounded by physical stall current (1.4A per XL330 servo)
        clamped_torques = np.clip(active_torques, -self.cfg.stall_torque, self.cfg.stall_torque)
        motor_currents = np.clip(
            np.abs(clamped_torques) / max(self.cfg.torque_constant, 1e-4),
            0.0,
            self.cfg.max_stall_current,
        )
        total_current = float(np.sum(motor_currents)) + self.cfg.idle_current
        self.last_current = total_current

        # Voltage sag: V_batt = V_oc - I * R_int
        v_drop = total_current * self.cfg.r_internal_batt
        v_terminal = max(self.cfg.v_min, self.cfg.v_open_circuit - v_drop)
        self.battery_voltage = v_terminal
        return v_terminal

    def compute_torques(
        self,
        target_positions: np.ndarray,
        measured_positions: np.ndarray,
        measured_velocities: np.ndarray,
        advance_delay: bool = True,
    ) -> np.ndarray:
        """Calculate actuator torques according to BAM M6 dynamics.

        Parameters
        ----------
        target_positions : np.ndarray
            Commanded joint positions from policy (14D).
        measured_positions : np.ndarray
            Encoder joint positions after backlash play (14D).
        measured_velocities : np.ndarray
            Encoder joint velocities after backlash play (14D).
        advance_delay : bool
            Whether to advance the delay queue during this call.

        Returns
        -------
        np.ndarray
            Torques to apply to the simulated joints (14D).
        """
        # 1. Transport delay
        if advance_delay:
            self._delay_queue.append(target_positions.copy())
            delayed_targets = self._delay_queue.popleft()
        else:
            delayed_targets = target_positions

        # 2. Firmware PD loop
        pos_error = delayed_targets - measured_positions
        raw_torques = self.cfg.kp * pos_error - self.cfg.kd * measured_velocities

        # 3. Battery voltage sag based on physical torque demand
        v_batt = self.update_battery_voltage(raw_torques)

        # 4. Back-EMF and velocity-dependent torque limits
        v_emf = self.cfg.back_emf_constant * np.abs(measured_velocities)
        v_effective = np.maximum(0.0, v_batt - v_emf)

        # Torque limit scaled by available effective voltage
        tau_max = self.cfg.stall_torque * (v_effective / self.cfg.v_nominal)
        self.last_torque_limits = tau_max.astype(np.float32)

        # 5. Gearbox friction and stiction
        net_torques = np.zeros_like(raw_torques)
        for i in range(self.num_actuators):
            vel = measured_velocities[i]
            tau_cmd = raw_torques[i]

            # Stiction: if near zero velocity and torque is below stiction, servo does not budge
            if abs(vel) < self.cfg.stiction_vel_threshold and abs(tau_cmd) < self.cfg.stiction_torque:
                net_torques[i] = 0.0
                continue

            # Coulomb friction opposes direction of torque/motion
            sign_tau = 1.0 if tau_cmd >= 0.0 else -1.0
            coulomb_term = max(0.0, abs(tau_cmd) - self.cfg.coulomb_friction)
            viscous_term = self.cfg.viscous_damping * vel

            tau_net = (sign_tau * coulomb_term) - viscous_term

            # Clamp to BAM effective maximum torque
            limit = tau_max[i]
            net_torques[i] = np.clip(tau_net, -limit, limit)

        self.last_torques = net_torques.astype(np.float32)
        return self.last_torques

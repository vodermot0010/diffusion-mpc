"""
Interaction-aware MPC refiner for Diffusion-Planner.

This module borrows the core idea from IJP: refine the ego trajectory while
jointly optimizing a small set of neighboring agents, keeping them close to the
learned diffusion prior instead of treating them as fixed obstacles.

Unlike the full IJP pipeline, this refiner is a lightweight post-processing
module that runs directly on top of Diffusion-Planner's ego-centric outputs.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import casadi as ca
import numpy as np


@dataclass
class MPCConfig:
    """MPC configuration parameters."""

    # Horizon
    horizon: int = 8
    dt: float = 0.1
    max_neighbors: int = 2
    activation_distance: float = 10.0
    min_ego_speed: float = 1.5
    interaction_clearance_threshold: float = 5.0
    min_clearance_improvement: float = 0.2
    refine_blend: float = 0.35
    refine_steps: int = 4
    max_position_delta: float = 1.0
    dense_neighbor_distance: float = 8.0
    dense_neighbor_count: int = 2
    dense_scene_clearance_threshold: float = 6.0
    dense_scene_min_clearance_improvement: float = 0.05
    dense_scene_refine_blend: float = 0.45
    dense_scene_refine_steps: int = 5
    static_obstacle_distance: float = 6.0
    static_obstacle_skip_count: int = 2

    # Ego vehicle parameters
    wheelbase: float = 2.89
    max_steer: float = 0.7
    max_accel: float = 3.0
    max_decel: float = -5.0
    max_speed: float = 20.0

    # Neighbor dynamics bounds
    max_neighbor_yaw_rate: float = 0.8
    max_neighbor_accel: float = 2.5
    max_neighbor_decel: float = -4.0
    max_neighbor_speed: float = 20.0

    # Safety parameters
    safety_margin: float = 3.0

    # Cost weights
    w_tracking: float = 2.0
    w_heading: float = 0.2
    w_speed: float = 0.1
    w_smooth: float = 0.1
    w_control: float = 0.02
    w_progress: float = 0.1
    w_safety: float = 40.0

    # Joint optimization weights
    w_neighbor_prior: float = 0.5
    w_neighbor_heading: float = 0.05
    w_neighbor_speed: float = 0.05
    w_neighbor_smooth: float = 0.02
    w_neighbor_control: float = 0.005

    # Solver settings
    max_iter: int = 50
    tol: float = 1e-4
    verbose: bool = False
    rebuild_after_failures: int = 2
    disable_after_failures: int = 8
    max_logged_failures: int = 3


class MPCRefiner:
    """Joint MPC refinement in the ego-centric frame."""

    def __init__(self, config: Optional[MPCConfig] = None):
        self.config = config or MPCConfig()
        self._opti = None
        self._vars = {}
        self._consecutive_failures = 0
        self._logged_failures = 0
        self._disabled = False

    def __getstate__(self):
        state = self.__dict__.copy()
        # CasADi Opti / SWIG-backed objects are not pickleable.
        state["_opti"] = None
        state["_vars"] = {}
        state["_consecutive_failures"] = 0
        state["_logged_failures"] = 0
        state["_disabled"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_optimizer(self):
        if self._opti is None:
            self._build_optimizer()

    def _release_optimizer(self):
        self._opti = None
        self._vars = {}

    @staticmethod
    def _heading_error(a, b):
        return ca.atan2(ca.sin(a - b), ca.cos(a - b))

    @staticmethod
    def _wrap_angle_np(angle: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _ego_dynamics(self, x, y, theta, v, delta, accel):
        dt = self.config.dt
        x_next = x + v * ca.cos(theta) * dt
        y_next = y + v * ca.sin(theta) * dt
        theta_next = theta + (v / self.config.wheelbase) * ca.tan(delta) * dt
        v_next = v + accel * dt
        return x_next, y_next, theta_next, v_next

    def _neighbor_dynamics(self, x, y, theta, v, yaw_rate, accel):
        dt = self.config.dt
        x_next = x + v * ca.cos(theta) * dt
        y_next = y + v * ca.sin(theta) * dt
        theta_next = theta + yaw_rate * dt
        v_next = v + accel * dt
        return x_next, y_next, theta_next, v_next

    def _build_optimizer(self):
        opti = ca.Opti()
        cfg = self.config
        N = cfg.horizon
        M = cfg.max_neighbors

        ego_x = opti.variable(N + 1)
        ego_y = opti.variable(N + 1)
        ego_theta = opti.variable(N + 1)
        ego_v = opti.variable(N + 1)
        ego_delta = opti.variable(N)
        ego_accel = opti.variable(N)

        neighbor_x = [opti.variable(N + 1) for _ in range(M)]
        neighbor_y = [opti.variable(N + 1) for _ in range(M)]
        neighbor_theta = [opti.variable(N + 1) for _ in range(M)]
        neighbor_v = [opti.variable(N + 1) for _ in range(M)]
        neighbor_yaw_rate = [opti.variable(N) for _ in range(M)]
        neighbor_accel = [opti.variable(N) for _ in range(M)]
        safety_slack = opti.variable(M, N + 1)

        ego_x0 = opti.parameter(4)
        ego_ref_x = opti.parameter(N + 1)
        ego_ref_y = opti.parameter(N + 1)
        ego_ref_theta = opti.parameter(N + 1)
        ego_ref_v = opti.parameter(N + 1)

        neighbor_mask = opti.parameter(M)
        neighbor_x0 = [opti.parameter(4) for _ in range(M)]
        neighbor_ref_x = [opti.parameter(N + 1) for _ in range(M)]
        neighbor_ref_y = [opti.parameter(N + 1) for _ in range(M)]
        neighbor_ref_theta = [opti.parameter(N + 1) for _ in range(M)]
        neighbor_ref_v = [opti.parameter(N + 1) for _ in range(M)]

        opti.subject_to(ego_x[0] == ego_x0[0])
        opti.subject_to(ego_y[0] == ego_x0[1])
        opti.subject_to(ego_theta[0] == ego_x0[2])
        opti.subject_to(ego_v[0] == ego_x0[3])

        for t in range(N):
            x_next, y_next, theta_next, v_next = self._ego_dynamics(
                ego_x[t], ego_y[t], ego_theta[t], ego_v[t], ego_delta[t], ego_accel[t]
            )
            opti.subject_to(ego_x[t + 1] == x_next)
            opti.subject_to(ego_y[t + 1] == y_next)
            opti.subject_to(ego_theta[t + 1] == theta_next)
            opti.subject_to(ego_v[t + 1] == v_next)
            opti.subject_to(opti.bounded(-cfg.max_steer, ego_delta[t], cfg.max_steer))
            opti.subject_to(opti.bounded(cfg.max_decel, ego_accel[t], cfg.max_accel))
            opti.subject_to(opti.bounded(0.0, ego_v[t], cfg.max_speed))
        opti.subject_to(opti.bounded(0.0, ego_v[N], cfg.max_speed))

        for n in range(M):
            opti.subject_to(neighbor_x[n][0] == neighbor_x0[n][0])
            opti.subject_to(neighbor_y[n][0] == neighbor_x0[n][1])
            opti.subject_to(neighbor_theta[n][0] == neighbor_x0[n][2])
            opti.subject_to(neighbor_v[n][0] == neighbor_x0[n][3])

            for t in range(N):
                x_next, y_next, theta_next, v_next = self._neighbor_dynamics(
                    neighbor_x[n][t],
                    neighbor_y[n][t],
                    neighbor_theta[n][t],
                    neighbor_v[n][t],
                    neighbor_yaw_rate[n][t],
                    neighbor_accel[n][t],
                )
                opti.subject_to(neighbor_x[n][t + 1] == x_next)
                opti.subject_to(neighbor_y[n][t + 1] == y_next)
                opti.subject_to(neighbor_theta[n][t + 1] == theta_next)
                opti.subject_to(neighbor_v[n][t + 1] == v_next)
                opti.subject_to(
                    opti.bounded(
                        -cfg.max_neighbor_yaw_rate,
                        neighbor_yaw_rate[n][t],
                        cfg.max_neighbor_yaw_rate,
                    )
                )
                opti.subject_to(
                    opti.bounded(
                        cfg.max_neighbor_decel,
                        neighbor_accel[n][t],
                        cfg.max_neighbor_accel,
                    )
                )
                opti.subject_to(opti.bounded(0.0, neighbor_v[n][t], cfg.max_neighbor_speed))
            opti.subject_to(opti.bounded(0.0, neighbor_v[n][N], cfg.max_neighbor_speed))

        cost = 0

        for t in range(N + 1):
            cost += cfg.w_tracking * (
                (ego_x[t] - ego_ref_x[t]) ** 2 + (ego_y[t] - ego_ref_y[t]) ** 2
            )
            cost += cfg.w_heading * self._heading_error(ego_theta[t], ego_ref_theta[t]) ** 2
            cost += cfg.w_speed * (ego_v[t] - ego_ref_v[t]) ** 2

            for n in range(M):
                mask = neighbor_mask[n]
                cost += cfg.w_neighbor_prior * mask * (
                    (neighbor_x[n][t] - neighbor_ref_x[n][t]) ** 2
                    + (neighbor_y[n][t] - neighbor_ref_y[n][t]) ** 2
                )
                cost += cfg.w_neighbor_heading * mask * self._heading_error(
                    neighbor_theta[n][t], neighbor_ref_theta[n][t]
                ) ** 2
                cost += cfg.w_neighbor_speed * mask * (
                    neighbor_v[n][t] - neighbor_ref_v[n][t]
                ) ** 2

                dist_sq = (ego_x[t] - neighbor_x[n][t]) ** 2 + (ego_y[t] - neighbor_y[n][t]) ** 2
                opti.subject_to(safety_slack[n, t] >= 0.0)
                opti.subject_to(
                    dist_sq + safety_slack[n, t] >= (cfg.safety_margin**2) * mask
                )
                cost += cfg.w_safety * safety_slack[n, t] ** 2

        for t in range(1, N + 1):
            cost += cfg.w_smooth * (
                (ego_x[t] - ego_x[t - 1]) ** 2 + (ego_y[t] - ego_y[t - 1]) ** 2
            )
            for n in range(M):
                mask = neighbor_mask[n]
                cost += cfg.w_neighbor_smooth * mask * (
                    (neighbor_x[n][t] - neighbor_x[n][t - 1]) ** 2
                    + (neighbor_y[n][t] - neighbor_y[n][t - 1]) ** 2
                )

        for t in range(N):
            cost += cfg.w_control * (ego_delta[t] ** 2 + ego_accel[t] ** 2)
            for n in range(M):
                mask = neighbor_mask[n]
                cost += cfg.w_neighbor_control * mask * (
                    neighbor_yaw_rate[n][t] ** 2 + neighbor_accel[n][t] ** 2
                )

        cost -= cfg.w_progress * (ego_x[N] - ego_x[0])
        opti.minimize(cost)

        opts = {
            "ipopt.print_level": 0 if not cfg.verbose else 5,
            "ipopt.max_iter": cfg.max_iter,
            "ipopt.tol": cfg.tol,
            "ipopt.acceptable_tol": max(cfg.tol * 10, 1e-3),
            "ipopt.acceptable_iter": 5,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.warm_start_init_point": "yes",
            "print_time": cfg.verbose,
            "ipopt.sb": "yes",
        }
        opti.solver("ipopt", opts)

        self._opti = opti
        self._vars = {
            "ego_x": ego_x,
            "ego_y": ego_y,
            "ego_theta": ego_theta,
            "ego_v": ego_v,
            "ego_delta": ego_delta,
            "ego_accel": ego_accel,
            "neighbor_x": neighbor_x,
            "neighbor_y": neighbor_y,
            "neighbor_theta": neighbor_theta,
            "neighbor_v": neighbor_v,
            "neighbor_yaw_rate": neighbor_yaw_rate,
            "neighbor_accel": neighbor_accel,
            "ego_x0": ego_x0,
            "ego_ref_x": ego_ref_x,
            "ego_ref_y": ego_ref_y,
            "ego_ref_theta": ego_ref_theta,
            "ego_ref_v": ego_ref_v,
            "neighbor_mask": neighbor_mask,
            "neighbor_x0": neighbor_x0,
            "neighbor_ref_x": neighbor_ref_x,
            "neighbor_ref_y": neighbor_ref_y,
            "neighbor_ref_theta": neighbor_ref_theta,
            "neighbor_ref_v": neighbor_ref_v,
            "safety_slack": safety_slack,
        }

    def _pad_to_horizon(self, values: np.ndarray, target_len: int) -> np.ndarray:
        if values.shape[0] >= target_len:
            return values[:target_len].copy()
        padded = np.zeros((target_len, values.shape[1]), dtype=np.float64)
        padded[: values.shape[0]] = values
        padded[values.shape[0] :] = values[-1]
        return padded

    def _future_prediction_to_reference(
        self, current_state: np.ndarray, future_prediction: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        future_xy = future_prediction[:, :2].astype(np.float64)
        future_theta = np.arctan2(future_prediction[:, 3], future_prediction[:, 2]).astype(np.float64)

        if future_xy.shape[0] == 0:
            future_v = np.zeros(0, dtype=np.float64)
        else:
            prev_xy = np.concatenate([current_state[None, :2], future_xy[:-1]], axis=0)
            future_v = np.linalg.norm(future_xy - prev_xy, axis=-1) / self.config.dt

        ref = np.zeros((future_prediction.shape[0] + 1, 4), dtype=np.float64)
        ref[0] = current_state
        if future_prediction.shape[0] > 0:
            ref[1:, 0:2] = future_xy
            ref[1:, 2] = future_theta
            ref[1:, 3] = future_v

        ref = self._pad_to_horizon(ref, self.config.horizon + 1)
        return ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]

    def _extract_relevant_neighbor_states(
        self, neighbor_current_states: np.ndarray, neighbor_predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_neighbors = self.config.max_neighbors
        current_dim = 4
        mask = np.zeros(max_neighbors, dtype=np.float64)
        states = np.zeros((max_neighbors, current_dim), dtype=np.float64)
        selected_predictions = np.zeros(
            (max_neighbors, neighbor_predictions.shape[1], neighbor_predictions.shape[2]),
            dtype=np.float64,
        )

        if neighbor_predictions.ndim != 3:
            return states, mask, selected_predictions

        candidates = []
        usable = min(neighbor_predictions.shape[0], neighbor_current_states.shape[0])
        for idx in range(usable):
            current_state = neighbor_current_states[idx].astype(np.float64)
            future_prediction = neighbor_predictions[idx].astype(np.float64)
            is_valid = bool(np.any(current_state[:2] != 0.0) or np.any(future_prediction[:, :2] != 0.0))
            if not is_valid:
                continue
            distance = float(np.linalg.norm(current_state[:2]))
            candidates.append((distance, current_state, future_prediction))

        if not candidates:
            return states, mask, selected_predictions

        candidates.sort(key=lambda item: item[0])
        kept = 0
        for distance, current_state, future_prediction in candidates:
            if kept >= max_neighbors:
                break
            if distance > self.config.activation_distance:
                continue
            states[kept] = current_state
            selected_predictions[kept] = future_prediction
            mask[kept] = 1.0
            kept += 1

        return states, mask, selected_predictions

    def _count_close_neighbors(self, neighbor_states: np.ndarray, neighbor_mask: np.ndarray, distance_threshold: float) -> int:
        count = 0
        for idx in np.where(neighbor_mask > 0.5)[0]:
            if float(np.linalg.norm(neighbor_states[idx, :2])) <= distance_threshold:
                count += 1
        return count

    def _count_close_static_obstacles(self, static_objects: Optional[np.ndarray], distance_threshold: float) -> int:
        if static_objects is None or static_objects.size == 0:
            return 0

        count = 0
        for obj in static_objects:
            if obj.shape[0] < 2:
                continue
            if not np.any(obj[:2] != 0.0):
                continue
            if float(np.linalg.norm(obj[:2])) <= distance_threshold:
                count += 1
        return count

    def _compose_reference_trajectory(self, current_state: np.ndarray, future_prediction: np.ndarray) -> np.ndarray:
        horizon = self.config.horizon + 1
        ref = np.zeros((horizon, 4), dtype=np.float64)
        ref_x, ref_y, ref_theta, ref_v = self._future_prediction_to_reference(current_state, future_prediction)
        ref[:, 0] = ref_x
        ref[:, 1] = ref_y
        ref[:, 2] = ref_theta
        ref[:, 3] = ref_v
        return ref

    def _estimate_ego_control_initial_guess(self, ref_theta: np.ndarray, ref_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        theta_delta = self._wrap_angle_np(ref_theta[1:] - ref_theta[:-1])
        denom = np.maximum(ref_v[:-1], 0.5) * cfg.dt
        steer = np.arctan(cfg.wheelbase * theta_delta / denom)
        steer = np.clip(steer, -cfg.max_steer, cfg.max_steer)
        accel = np.clip((ref_v[1:] - ref_v[:-1]) / cfg.dt, cfg.max_decel, cfg.max_accel)
        return steer.astype(np.float64), accel.astype(np.float64)

    def _estimate_neighbor_control_initial_guess(
        self, ref_theta: np.ndarray, ref_v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        theta_delta = self._wrap_angle_np(ref_theta[1:] - ref_theta[:-1])
        yaw_rate = np.clip(theta_delta / cfg.dt, -cfg.max_neighbor_yaw_rate, cfg.max_neighbor_yaw_rate)
        accel = np.clip((ref_v[1:] - ref_v[:-1]) / cfg.dt, cfg.max_neighbor_decel, cfg.max_neighbor_accel)
        return yaw_rate.astype(np.float64), accel.astype(np.float64)

    def _min_clearance(self, ego_traj: np.ndarray, neighbor_trajs: np.ndarray, neighbor_mask: np.ndarray) -> float:
        active_indices = np.where(neighbor_mask > 0.5)[0]
        if len(active_indices) == 0:
            return np.inf

        min_clearance = np.inf
        for idx in active_indices:
            diff = ego_traj[:, :2] - neighbor_trajs[idx, :, :2]
            dists = np.linalg.norm(diff, axis=-1)
            min_clearance = min(min_clearance, float(np.min(dists)))
        return min_clearance

    def _blend_heading(self, base: np.ndarray, refined: np.ndarray, alpha: float) -> np.ndarray:
        blended_cos = (1.0 - alpha) * base[:, 2] + alpha * refined[:, 2]
        blended_sin = (1.0 - alpha) * base[:, 3] + alpha * refined[:, 3]
        norm = np.maximum(np.sqrt(blended_cos**2 + blended_sin**2), 1e-6)
        return np.stack([blended_cos / norm, blended_sin / norm], axis=-1)

    def refine(
        self,
        ego_state: np.ndarray,
        ego_prediction: np.ndarray,
        neighbor_current_states: np.ndarray,
        neighbor_predictions: np.ndarray,
        static_objects: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Refine diffusion outputs in the ego-centric frame."""
        cfg = self.config
        M = cfg.max_neighbors
        N = cfg.horizon

        if self._disabled:
            return ego_prediction, neighbor_predictions

        if ego_state.shape[0] > 3 and float(ego_state[3]) < cfg.min_ego_speed:
            return ego_prediction, neighbor_predictions

        ego_ref_x, ego_ref_y, ego_ref_theta, ego_ref_v = self._future_prediction_to_reference(
            ego_state, ego_prediction
        )

        valid_neighbor_states, neighbor_mask, selected_neighbor_predictions = self._extract_relevant_neighbor_states(
            neighbor_current_states, neighbor_predictions
        )

        if np.sum(neighbor_mask) == 0:
            return ego_prediction, neighbor_predictions

        close_neighbor_count = self._count_close_neighbors(
            valid_neighbor_states, neighbor_mask, cfg.dense_neighbor_distance
        )
        close_static_count = self._count_close_static_obstacles(static_objects, cfg.static_obstacle_distance)
        if close_static_count >= cfg.static_obstacle_skip_count and close_neighbor_count < cfg.dense_neighbor_count:
            return ego_prediction, neighbor_predictions

        is_dense_dynamic_scene = close_neighbor_count >= cfg.dense_neighbor_count
        prior_ego_traj = self._compose_reference_trajectory(ego_state, ego_prediction)
        prior_neighbor_trajs = np.zeros((M, cfg.horizon + 1, 4), dtype=np.float64)
        for idx in range(M):
            prior_neighbor_trajs[idx] = self._compose_reference_trajectory(
                valid_neighbor_states[idx], selected_neighbor_predictions[idx]
            )

        prior_min_clearance = self._min_clearance(prior_ego_traj, prior_neighbor_trajs, neighbor_mask)
        activation_clearance_threshold = (
            cfg.dense_scene_clearance_threshold if is_dense_dynamic_scene else cfg.interaction_clearance_threshold
        )
        if prior_min_clearance > activation_clearance_threshold:
            return ego_prediction, neighbor_predictions

        self._ensure_optimizer()

        self._opti.set_value(self._vars["ego_x0"], ego_state)
        self._opti.set_value(self._vars["ego_ref_x"], ego_ref_x)
        self._opti.set_value(self._vars["ego_ref_y"], ego_ref_y)
        self._opti.set_value(self._vars["ego_ref_theta"], ego_ref_theta)
        self._opti.set_value(self._vars["ego_ref_v"], ego_ref_v)
        self._opti.set_value(self._vars["neighbor_mask"], neighbor_mask)

        ego_delta_init, ego_accel_init = self._estimate_ego_control_initial_guess(ego_ref_theta, ego_ref_v)
        self._opti.set_initial(self._vars["ego_x"], ego_ref_x)
        self._opti.set_initial(self._vars["ego_y"], ego_ref_y)
        self._opti.set_initial(self._vars["ego_theta"], ego_ref_theta)
        self._opti.set_initial(self._vars["ego_v"], ego_ref_v)
        self._opti.set_initial(self._vars["ego_delta"], ego_delta_init)
        self._opti.set_initial(self._vars["ego_accel"], ego_accel_init)

        neighbor_ref_x_all = np.zeros((M, N + 1), dtype=np.float64)
        neighbor_ref_y_all = np.zeros((M, N + 1), dtype=np.float64)

        for idx in range(M):
            current_state = valid_neighbor_states[idx]
            self._opti.set_value(self._vars["neighbor_x0"][idx], current_state)

            if neighbor_mask[idx] > 0.0:
                ref_x, ref_y, ref_theta, ref_v = self._future_prediction_to_reference(
                    current_state, selected_neighbor_predictions[idx]
                )
            else:
                zeros = np.zeros(N + 1, dtype=np.float64)
                ref_x = np.full(N + 1, current_state[0], dtype=np.float64)
                ref_y = np.full(N + 1, current_state[1], dtype=np.float64)
                ref_theta = np.full(N + 1, current_state[2], dtype=np.float64)
                ref_v = zeros

            self._opti.set_value(self._vars["neighbor_ref_x"][idx], ref_x)
            self._opti.set_value(self._vars["neighbor_ref_y"][idx], ref_y)
            self._opti.set_value(self._vars["neighbor_ref_theta"][idx], ref_theta)
            self._opti.set_value(self._vars["neighbor_ref_v"][idx], ref_v)

            neighbor_ref_x_all[idx] = ref_x
            neighbor_ref_y_all[idx] = ref_y
            neighbor_yaw_rate_init, neighbor_accel_init = self._estimate_neighbor_control_initial_guess(ref_theta, ref_v)
            self._opti.set_initial(self._vars["neighbor_x"][idx], ref_x)
            self._opti.set_initial(self._vars["neighbor_y"][idx], ref_y)
            self._opti.set_initial(self._vars["neighbor_theta"][idx], ref_theta)
            self._opti.set_initial(self._vars["neighbor_v"][idx], ref_v)
            self._opti.set_initial(self._vars["neighbor_yaw_rate"][idx], neighbor_yaw_rate_init)
            self._opti.set_initial(self._vars["neighbor_accel"][idx], neighbor_accel_init)

        safety_slack_init = np.zeros((M, N + 1), dtype=np.float64)
        for idx in range(M):
            if neighbor_mask[idx] <= 0.0:
                continue
            dist_sq = (ego_ref_x - neighbor_ref_x_all[idx]) ** 2 + (ego_ref_y - neighbor_ref_y_all[idx]) ** 2
            safety_slack_init[idx] = np.maximum(0.0, cfg.safety_margin**2 - dist_sq)
        self._opti.set_initial(self._vars["safety_slack"], safety_slack_init)

        try:
            sol = self._opti.solve()
        except RuntimeError as exc:
            failure_text = str(exc)
            self._consecutive_failures += 1
            if self._logged_failures < cfg.max_logged_failures:
                print(f"MPC optimization failed: {exc}, returning original diffusion output")
                self._logged_failures += 1
            elif self._logged_failures == cfg.max_logged_failures:
                print("MPC optimization keeps failing; suppressing further logs for this worker")
                self._logged_failures += 1

            if self._consecutive_failures >= cfg.rebuild_after_failures:
                self._release_optimizer()

            if "Maximum_Iterations_Exceeded" in failure_text:
                if self._consecutive_failures >= cfg.rebuild_after_failures:
                    self._consecutive_failures = 0
                return ego_prediction, neighbor_predictions

            if self._consecutive_failures >= cfg.disable_after_failures:
                self._disabled = True
                self._release_optimizer()
                print("MPC disabled for this worker after repeated solver failures")
            return ego_prediction, neighbor_predictions

        self._consecutive_failures = 0

        refined_ego = ego_prediction.copy()
        ego_steps = min(N, ego_prediction.shape[0])
        refined_ego[:ego_steps, 0] = np.asarray(sol.value(self._vars["ego_x"]))[1 : ego_steps + 1]
        refined_ego[:ego_steps, 1] = np.asarray(sol.value(self._vars["ego_y"]))[1 : ego_steps + 1]
        refined_theta = np.asarray(sol.value(self._vars["ego_theta"]))[1 : ego_steps + 1]
        refined_ego[:ego_steps, 2] = np.cos(refined_theta)
        refined_ego[:ego_steps, 3] = np.sin(refined_theta)

        refined_neighbors = neighbor_predictions.copy()
        writeback_indices = []
        usable = min(neighbor_predictions.shape[0], neighbor_current_states.shape[0])
        for idx in range(usable):
            if not (np.any(neighbor_current_states[idx, :2] != 0.0) or np.any(neighbor_predictions[idx, :, :2] != 0.0)):
                continue
            if float(np.linalg.norm(neighbor_current_states[idx, :2])) <= cfg.activation_distance:
                writeback_indices.append(idx)
            if len(writeback_indices) >= M:
                break

        for local_idx, original_idx in enumerate(writeback_indices):
            if neighbor_mask[local_idx] <= 0.0:
                continue
            neighbor_steps = min(N, neighbor_predictions[original_idx].shape[0])
            refined_neighbors[original_idx, :neighbor_steps, 0] = np.asarray(sol.value(self._vars["neighbor_x"][local_idx]))[
                1 : neighbor_steps + 1
            ]
            refined_neighbors[original_idx, :neighbor_steps, 1] = np.asarray(sol.value(self._vars["neighbor_y"][local_idx]))[
                1 : neighbor_steps + 1
            ]
            refined_theta = np.asarray(sol.value(self._vars["neighbor_theta"][local_idx]))[1 : neighbor_steps + 1]
            refined_neighbors[original_idx, :neighbor_steps, 2] = np.cos(refined_theta)
            refined_neighbors[original_idx, :neighbor_steps, 3] = np.sin(refined_theta)

        refined_ego_traj = self._compose_reference_trajectory(ego_state, refined_ego)
        refined_neighbor_trajs = np.zeros((M, cfg.horizon + 1, 4), dtype=np.float64)
        for idx in range(M):
            if neighbor_mask[idx] <= 0.0:
                continue
            refined_neighbor_trajs[idx] = self._compose_reference_trajectory(
                valid_neighbor_states[idx], selected_neighbor_predictions[idx]
            )
        for local_idx, original_idx in enumerate(writeback_indices):
            if local_idx >= M or neighbor_mask[local_idx] <= 0.0:
                continue
            refined_neighbor_trajs[local_idx] = self._compose_reference_trajectory(
                valid_neighbor_states[local_idx], refined_neighbors[original_idx]
            )

        refined_min_clearance = self._min_clearance(refined_ego_traj, refined_neighbor_trajs, neighbor_mask)
        required_clearance_improvement = (
            cfg.dense_scene_min_clearance_improvement
            if is_dense_dynamic_scene
            else cfg.min_clearance_improvement
        )
        if refined_min_clearance < prior_min_clearance + required_clearance_improvement:
            return ego_prediction, neighbor_predictions

        blend_alpha = cfg.dense_scene_refine_blend if is_dense_dynamic_scene else cfg.refine_blend
        blend_steps = min(
            cfg.dense_scene_refine_steps if is_dense_dynamic_scene else cfg.refine_steps,
            ego_steps,
        )
        if blend_steps > 0:
            position_delta = np.linalg.norm(
                refined_ego[:blend_steps, :2] - ego_prediction[:blend_steps, :2], axis=-1
            )
            if float(np.max(position_delta)) > cfg.max_position_delta:
                return ego_prediction, neighbor_predictions

            refined_ego[:blend_steps, :2] = (
                (1.0 - blend_alpha) * ego_prediction[:blend_steps, :2]
                + blend_alpha * refined_ego[:blend_steps, :2]
            )
            blended_heading = self._blend_heading(
                ego_prediction[:blend_steps], refined_ego[:blend_steps], blend_alpha
            )
            refined_ego[:blend_steps, 2] = blended_heading[:, 0]
            refined_ego[:blend_steps, 3] = blended_heading[:, 1]

        return refined_ego, refined_neighbors

    def refine_batch(
        self,
        ego_states: np.ndarray,
        ego_predictions: np.ndarray,
        neighbor_current_states: np.ndarray,
        neighbor_predictions: np.ndarray,
    ) -> np.ndarray:
        """Sequential batch refinement."""
        batch_size = ego_predictions.shape[0]
        refined = np.zeros_like(ego_predictions)

        for idx in range(batch_size):
            refined[idx], _ = self.refine(
                ego_states[idx],
                ego_predictions[idx],
                neighbor_current_states[idx],
                neighbor_predictions[idx],
            )

        return refined


class MPCRefinerTorch(MPCRefiner):
    """PyTorch-compatible wrapper for MPCRefiner."""

    def refine_torch(
        self,
        ego_state: "torch.Tensor",
        ego_prediction: "torch.Tensor",
        neighbor_current_states: "torch.Tensor",
        neighbor_predictions: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        import torch

        if ego_state.dim() == 2:
            ego_state = ego_state[0]
        if ego_prediction.dim() == 3:
            ego_prediction = ego_prediction[0]
        if neighbor_current_states.dim() == 3:
            neighbor_current_states = neighbor_current_states[0]
        if neighbor_predictions.dim() == 4:
            neighbor_predictions = neighbor_predictions[0]

        refined_ego_np, refined_neighbors_np = self.refine(
            ego_state.detach().cpu().numpy(),
            ego_prediction.detach().cpu().numpy(),
            neighbor_current_states.detach().cpu().numpy(),
            neighbor_predictions.detach().cpu().numpy(),
        )

        refined_ego = torch.from_numpy(refined_ego_np).float().to(ego_prediction.device)
        refined_neighbors = torch.from_numpy(refined_neighbors_np).float().to(
            neighbor_predictions.device
        )
        return refined_ego, refined_neighbors

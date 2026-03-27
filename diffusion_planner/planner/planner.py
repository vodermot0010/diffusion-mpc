
import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type, Optional

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.utils.config import Config
from diffusion_planner.optimization.mpc_refiner import MPCRefiner, MPCConfig


def identity(ego_state, predictions):
    return predictions


class DiffusionPlanner(AbstractPlanner):
    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling,
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",

            # Optional interaction-aware MPC refinement.
            mpc_enabled: bool = False,
            mpc_config: Optional[Dict] = None,
            candidate_sampling: Optional[Dict] = None,
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"

        self._future_horizon = future_trajectory_sampling.time_horizon
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses

        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        self._planner = Diffusion_Planner(config)

        self.data_processor = DataProcessor(config)
        self.observation_normalizer = config.observation_normalizer

        self._mpc_enabled = mpc_enabled
        self._mpc_config_dict = mpc_config or {}
        self._mpc_refiner = None
        self._candidate_sampling = candidate_sampling or {"enabled": False}

    def __getstate__(self):
        state = self.__dict__.copy()
        # The lazy MPC refiner owns CasADi/SWIG objects that cannot be pickled
        # by nuPlan's simulation log callback.
        state["_mpc_refiner"] = None
        return state

    def name(self) -> str:
        return "diffusion_planner"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        if self._ckpt_path is not None:
            state_dict: Dict = torch.load(self._ckpt_path, map_location=self._device)

            if self._ema_enabled:
                state_dict = state_dict["ema_state_dict"]
            else:
                if "model" in state_dict.keys():
                    state_dict = state_dict["model"]
            model_state_dict = {
                k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")
            }
            self._planner.load_state_dict(model_state_dict)
        else:
            print("load random model")

        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(
            history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device
        )
        return model_inputs

    def outputs_to_trajectory(
        self,
        outputs: Dict[str, torch.Tensor],
        ego_state_history: Deque[EgoState],
    ) -> List[InterpolatableState]:
        predictions = outputs["prediction"][0, 0].detach().cpu().numpy().astype(np.float64)
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1)

        states = transform_predictions_to_states(
            predictions,
            ego_state_history,
            self._future_horizon,
            self._step_interval,
        )
        return states

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.

        Pipeline:
        1. Diffusion-Planner generates an ego-centric joint future.
        2. Optional MPC refiner improves the first part of the rollout in the same local frame.
        """
        raw_inputs = self.planner_input_to_model_inputs(current_input)
        normalized_inputs = self.observation_normalizer(raw_inputs)
        outputs = self._run_diffusion_inference(normalized_inputs, raw_inputs, current_input)

        if self._mpc_enabled:
            outputs = self._apply_mpc_refinement(outputs, raw_inputs, current_input)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )
        return trajectory

    def _run_model_once(self, normalized_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            _, outputs = self._planner(normalized_inputs)
        return outputs

    def _current_neighbor_states(self, raw_inputs: Dict[str, torch.Tensor], predicted_neighbor_num: int) -> np.ndarray:
        neighbor_current = raw_inputs["neighbor_agents_past"][0, :predicted_neighbor_num, -1]
        neighbor_current = neighbor_current.detach().cpu().numpy()
        neighbor_heading = np.arctan2(neighbor_current[:, 3], neighbor_current[:, 2])
        neighbor_speed = np.linalg.norm(neighbor_current[:, 4:6], axis=-1)
        return np.stack(
            [neighbor_current[:, 0], neighbor_current[:, 1], neighbor_heading, neighbor_speed],
            axis=-1,
        ).astype(np.float64)

    def _count_close_points(self, points: np.ndarray, threshold: float) -> int:
        if points.size == 0:
            return 0
        count = 0
        for point in points:
            if point.shape[0] < 2:
                continue
            if not np.any(point[:2] != 0.0):
                continue
            if float(np.linalg.norm(point[:2])) <= threshold:
                count += 1
        return count

    def _scene_context(self, raw_inputs: Dict[str, torch.Tensor], predicted_neighbor_num: int) -> Dict[str, float]:
        cfg = self._candidate_sampling
        neighbor_states = self._current_neighbor_states(raw_inputs, predicted_neighbor_num)
        static_objects = raw_inputs["static_objects"][0].detach().cpu().numpy().astype(np.float64)

        close_dynamic_count = self._count_close_points(
            neighbor_states, float(cfg.get("risk_neighbor_distance", 8.0))
        )
        close_static_count = self._count_close_points(
            static_objects, float(cfg.get("risk_static_distance", 5.0))
        )

        lead_distance = np.inf
        lead_lateral_threshold = float(cfg.get("lead_vehicle_lateral_threshold", 2.5))
        lead_heading_threshold = float(cfg.get("lead_vehicle_heading_threshold", 0.35))
        lead_distance_threshold = float(cfg.get("lead_vehicle_distance_threshold", 20.0))

        for state in neighbor_states:
            if not np.any(state[:2] != 0.0):
                continue
            x, y, heading, _ = state
            if x <= 0.0:
                continue
            if abs(y) > lead_lateral_threshold:
                continue
            if abs(heading) > lead_heading_threshold:
                continue
            distance = float(np.linalg.norm(state[:2]))
            if distance <= lead_distance_threshold:
                lead_distance = min(lead_distance, distance)

        lead_close_neighbor_count = 0
        for state in neighbor_states:
            if not np.any(state[:2] != 0.0):
                continue
            distance = float(np.linalg.norm(state[:2]))
            if distance <= float(cfg.get("lead_scene_neighbor_distance", 12.0)):
                lead_close_neighbor_count += 1

        is_following_lead = np.isfinite(lead_distance) and lead_close_neighbor_count <= int(
            cfg.get("lead_scene_max_neighbors", 1)
        )

        return {
            "neighbor_states": neighbor_states,
            "static_objects": static_objects,
            "close_dynamic_count": close_dynamic_count,
            "close_static_count": close_static_count,
            "lead_distance": lead_distance,
            "is_following_lead": bool(is_following_lead),
            "is_dense_dynamic_scene": close_dynamic_count >= int(cfg.get("risk_neighbor_count", 2)),
        }

    def _should_use_candidate_sampling(
        self,
        raw_inputs: Dict[str, torch.Tensor],
        current_input: PlannerInput,
        predicted_neighbor_num: int,
    ) -> bool:
        cfg = self._candidate_sampling
        if not cfg.get("enabled", False):
            return False

        ego_speed = float(current_input.history.ego_states[-1].dynamic_car_state.speed)
        if ego_speed < float(cfg.get("min_ego_speed", 1.0)):
            return False

        scene = self._scene_context(raw_inputs, predicted_neighbor_num)
        if scene["is_following_lead"] and scene["lead_distance"] > float(
            cfg.get("lead_vehicle_critical_distance", 8.0)
        ):
            return False

        return (
            scene["close_dynamic_count"] >= int(cfg.get("risk_neighbor_count", 2))
            or scene["close_static_count"] >= int(cfg.get("risk_static_count", 2))
        )

    def _candidate_route_deviation(self, ego_pred: np.ndarray, route_lanes: np.ndarray, steps: int) -> float:
        route_points = route_lanes[..., :2].reshape(-1, 2)
        valid = np.any(route_points != 0.0, axis=-1)
        route_points = route_points[valid]
        if route_points.shape[0] == 0:
            return 0.0

        ego_xy = ego_pred[:steps, :2]
        dists = np.linalg.norm(ego_xy[:, None, :] - route_points[None, :, :], axis=-1)
        return float(np.mean(np.min(dists, axis=-1)))

    def _candidate_min_clearance(self, ego_pred: np.ndarray, others: np.ndarray, steps: int) -> float:
        min_clearance = np.inf
        for other in others:
            if other.shape[0] == 0 or not np.any(other[:, :2] != 0.0):
                continue
            dists = np.linalg.norm(ego_pred[:steps, :2] - other[:steps, :2], axis=-1)
            min_clearance = min(min_clearance, float(np.min(dists)))
        return min_clearance

    def _candidate_static_clearance(self, ego_pred: np.ndarray, static_objects: np.ndarray, steps: int) -> float:
        valid = static_objects[np.any(static_objects[:, :2] != 0.0, axis=-1)]
        if valid.shape[0] == 0:
            return np.inf
        dists = np.linalg.norm(ego_pred[:steps, None, :2] - valid[None, :, :2], axis=-1)
        return float(np.min(dists))

    def _score_candidate(self, outputs: Dict[str, torch.Tensor], raw_inputs: Dict[str, torch.Tensor]) -> float:
        cfg = self._candidate_sampling
        scene = self._scene_context(raw_inputs, getattr(self._config, "predicted_neighbor_num", 10))
        prediction = outputs["prediction"][0].detach().cpu().numpy().astype(np.float64)
        ego_pred = prediction[0]
        neighbor_preds = prediction[1:]
        static_objects = scene["static_objects"]
        route_lanes = raw_inputs["route_lanes"][0].detach().cpu().numpy().astype(np.float64)
        steps = min(int(cfg.get("evaluate_steps", 8)), ego_pred.shape[0])
        if steps <= 1:
            return 0.0

        progress = float(ego_pred[steps - 1, 0])
        neighbor_clearance = self._candidate_min_clearance(ego_pred, neighbor_preds, steps)
        static_clearance = self._candidate_static_clearance(ego_pred, static_objects, steps)
        route_deviation = self._candidate_route_deviation(ego_pred, route_lanes, steps)
        smoothness = float(np.mean(np.linalg.norm(np.diff(ego_pred[:steps, :2], n=2, axis=0), axis=-1))) if steps > 2 else 0.0
        clearance_clip = float(cfg.get("clearance_clip", 10.0))

        if neighbor_clearance < float(cfg.get("hard_neighbor_clearance", 2.5)):
            return -1e6
        if static_clearance < float(cfg.get("hard_static_clearance", 1.5)):
            return -1e6

        score = 0.0
        score += float(cfg.get("progress_weight", 1.0)) * progress
        neighbor_clearance_weight = float(
            cfg.get(
                "dense_neighbor_clearance_weight" if scene["is_dense_dynamic_scene"] else "neighbor_clearance_weight",
                2.0 if not scene["is_dense_dynamic_scene"] else 2.5,
            )
        )
        score += neighbor_clearance_weight * min(neighbor_clearance, clearance_clip)
        score += float(cfg.get("static_clearance_weight", 1.5)) * min(static_clearance, clearance_clip)
        score -= float(cfg.get("route_weight", 0.5)) * route_deviation
        smooth_weight = float(
            cfg.get("lead_scene_smooth_weight", 0.35)
            if scene["is_following_lead"]
            else cfg.get("smooth_weight", 0.2)
        )
        score -= smooth_weight * smoothness
        return score

    def _run_diffusion_inference(
        self,
        normalized_inputs: Dict[str, torch.Tensor],
        raw_inputs: Dict[str, torch.Tensor],
        current_input: PlannerInput,
    ) -> Dict[str, torch.Tensor]:
        predicted_neighbor_num = getattr(self._config, "predicted_neighbor_num", 10)
        if not self._should_use_candidate_sampling(raw_inputs, current_input, predicted_neighbor_num):
            return self._run_model_once(normalized_inputs)

        scene = self._scene_context(raw_inputs, predicted_neighbor_num)
        num_candidates = int(
            self._candidate_sampling.get(
                "dense_scene_num_candidates" if scene["is_dense_dynamic_scene"] else "num_candidates",
                4 if scene["is_dense_dynamic_scene"] else 3,
            )
        )
        best_outputs = None
        best_score = -np.inf

        for _ in range(num_candidates):
            outputs = self._run_model_once(normalized_inputs)
            score = self._score_candidate(outputs, raw_inputs)
            if score > best_score:
                best_score = score
                best_outputs = outputs

        return best_outputs if best_outputs is not None else self._run_model_once(normalized_inputs)

    def _apply_mpc_refinement(
        self,
        outputs: Dict[str, torch.Tensor],
        raw_inputs: Dict[str, torch.Tensor],
        current_input: PlannerInput,
    ) -> Dict[str, torch.Tensor]:
        """
        Refine diffusion output with a lightweight joint MPC in the ego-centric frame.
        """
        if self._mpc_refiner is None:
            mpc_cfg = MPCConfig(**self._mpc_config_dict)
            self._mpc_refiner = MPCRefiner(config=mpc_cfg)

        predictions = outputs["prediction"]
        ego_pred = predictions[0, 0].detach().cpu().numpy()
        neighbor_preds = predictions[0, 1:].detach().cpu().numpy()
        predicted_neighbor_num = getattr(self._config, "predicted_neighbor_num", 10)
        scene = self._scene_context(raw_inputs, predicted_neighbor_num)
        if scene["is_following_lead"] and scene["lead_distance"] > float(
            self._candidate_sampling.get("lead_vehicle_critical_distance", 8.0)
        ):
            return outputs

        ego_speed = current_input.history.ego_states[-1].dynamic_car_state.speed
        ego_current = np.array([0.0, 0.0, 0.0, ego_speed], dtype=np.float64)

        neighbor_current = raw_inputs["neighbor_agents_past"][0, : neighbor_preds.shape[0], -1]
        neighbor_current = neighbor_current.detach().cpu().numpy()
        neighbor_heading = np.arctan2(neighbor_current[:, 3], neighbor_current[:, 2])
        neighbor_speed = np.linalg.norm(neighbor_current[:, 4:6], axis=-1)
        neighbor_current_states = np.stack(
            [neighbor_current[:, 0], neighbor_current[:, 1], neighbor_heading, neighbor_speed],
            axis=-1,
        ).astype(np.float64)
        static_objects = raw_inputs["static_objects"][0].detach().cpu().numpy().astype(np.float64)

        try:
            refined_ego, refined_neighbors = self._mpc_refiner.refine(
                ego_state=ego_current,
                ego_prediction=ego_pred,
                neighbor_current_states=neighbor_current_states,
                neighbor_predictions=neighbor_preds,
                static_objects=static_objects,
            )

            outputs["prediction"] = outputs["prediction"].clone()
            outputs["prediction"][0, 0] = torch.from_numpy(refined_ego).float().to(predictions.device)
            if refined_neighbors.shape[0] > 0:
                outputs["prediction"][0, 1 : 1 + refined_neighbors.shape[0]] = (
                    torch.from_numpy(refined_neighbors).float().to(predictions.device)
                )

        except Exception as exc:
            print(f"MPC refinement failed: {exc}, using original diffusion output")

        return outputs

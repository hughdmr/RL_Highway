from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym


class SafeDrivingRewardWrapper(gym.Wrapper):
    """Shape highway-env rewards toward safer, smoother driving.

    The wrapper keeps the original environment dynamics unchanged. It only
    reshapes the reward by:
    - penalizing crashes strongly,
    - rewarding survival slightly,
    - penalizing abrupt action switches as a proxy for jerky driving.
    - rewarding safe lane changes when a slower vehicle is ahead,
    - applying a small penalty when a safe pass was available but ignored.
    """

    def __init__(
        self,
        env: gym.Env,
        crash_penalty: float = 5.0,
        survival_bonus: float = 0.01,
        action_switch_penalty: float = 0.02,
        safe_lane_change_bonus: float = 0.15,
        missed_pass_penalty: float = 0.03,
        front_speed_margin: float = 2.0,
        front_gap_threshold: float = 25.0,
        adjacent_gap_threshold: float = 18.0,
    ):
        super().__init__(env)
        self.crash_penalty = float(crash_penalty)
        self.survival_bonus = float(survival_bonus)
        self.action_switch_penalty = float(action_switch_penalty)
        self.safe_lane_change_bonus = float(safe_lane_change_bonus)
        self.missed_pass_penalty = float(missed_pass_penalty)
        self.front_speed_margin = float(front_speed_margin)
        self.front_gap_threshold = float(front_gap_threshold)
        self.adjacent_gap_threshold = float(adjacent_gap_threshold)
        self._previous_action: Optional[int] = None
        self._action_indexes = {
            "LANE_LEFT": 0,
            "IDLE": 1,
            "LANE_RIGHT": 2,
            "FASTER": 3,
            "SLOWER": 4,
        }

    def _refresh_action_indexes(self) -> None:
        action_type = getattr(self.env.unwrapped, "action_type", None)
        if action_type is not None and hasattr(action_type, "actions_indexes"):
            self._action_indexes = dict(action_type.actions_indexes)

    def _ego_vehicle(self):
        return getattr(self.env.unwrapped, "vehicle", None)

    def _road(self):
        return getattr(self.env.unwrapped, "road", None)

    @staticmethod
    def _lane_id(lane_index) -> Optional[int]:
        if lane_index is None:
            return None
        try:
            return int(lane_index[2])
        except Exception:
            return None

    @staticmethod
    def _replace_lane_id(lane_index, lane_id: int):
        if lane_index is None:
            return None
        try:
            return (lane_index[0], lane_index[1], int(lane_id))
        except Exception:
            return None

    def _lane_count(self) -> int:
        config = getattr(self.env.unwrapped, "config", None)
        if isinstance(config, dict):
            return int(config.get("lanes_count", 4))
        return 4

    def _gap_to_vehicle(self, other_vehicle) -> float:
        ego = self._ego_vehicle()
        if ego is None or other_vehicle is None:
            return float("inf")
        return float(other_vehicle.position[0] - ego.position[0])

    def _lane_status(self, lane_index) -> Tuple[float, float, bool]:
        """Return (front_gap, rear_gap, lane_is_safe)."""
        road = self._road()
        ego = self._ego_vehicle()
        if road is None or ego is None or lane_index is None:
            return float("inf"), float("inf"), False

        front_vehicle, rear_vehicle = road.neighbour_vehicles(ego, lane_index=lane_index)
        front_gap = self._gap_to_vehicle(front_vehicle)
        rear_gap = -self._gap_to_vehicle(rear_vehicle)
        safe = front_gap > self.adjacent_gap_threshold and rear_gap > self.adjacent_gap_threshold
        return front_gap, rear_gap, safe

    def _slower_vehicle_ahead(self) -> bool:
        road = self._road()
        ego = self._ego_vehicle()
        if road is None or ego is None:
            return False

        front_vehicle, _ = road.neighbour_vehicles(ego)
        if front_vehicle is None:
            return False

        front_gap = self._gap_to_vehicle(front_vehicle)
        speed_diff = float(ego.speed - front_vehicle.speed)
        return front_gap < self.front_gap_threshold and speed_diff > self.front_speed_margin

    def _safe_pass_available(self) -> bool:
        ego = self._ego_vehicle()
        if ego is None:
            return False

        lane_id = self._lane_id(getattr(ego, "lane_index", None))
        if lane_id is None:
            return False

        road = self._road()
        if road is None:
            return False

        lanes_count = self._lane_count()
        for candidate_lane_id in (lane_id - 1, lane_id + 1):
            if candidate_lane_id < 0 or candidate_lane_id >= lanes_count:
                continue
            candidate_lane_index = self._replace_lane_id(getattr(ego, "lane_index", None), candidate_lane_id)
            _, _, safe = self._lane_status(candidate_lane_index)
            if safe:
                return True
        return False

    def _choose_lane_change_bonus(self, action: int) -> float:
        ego = self._ego_vehicle()
        if ego is None:
            return 0.0

        lane_id = self._lane_id(getattr(ego, "lane_index", None))
        if lane_id is None:
            return 0.0

        lanes_count = self._lane_count()
        left_action = self._action_indexes.get("LANE_LEFT", 0)
        right_action = self._action_indexes.get("LANE_RIGHT", 2)
        lane_left_safe = False
        lane_right_safe = False

        if lane_id - 1 >= 0:
            lane_left_safe = self._lane_status(self._replace_lane_id(getattr(ego, "lane_index", None), lane_id - 1))[2]
        if lane_id + 1 < lanes_count:
            lane_right_safe = self._lane_status(self._replace_lane_id(getattr(ego, "lane_index", None), lane_id + 1))[2]

        if action == left_action and lane_left_safe:
            return self.safe_lane_change_bonus
        if action == right_action and lane_right_safe:
            return self.safe_lane_change_bonus

        if self._slower_vehicle_ahead() and (lane_left_safe or lane_right_safe):
            if action not in (left_action, right_action):
                return -self.missed_pass_penalty

        return 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._previous_action = None
        self._refresh_action_indexes()
        info = dict(info)
        info["safety/wrapper_active"] = True
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = float(reward)
        info = dict(info)
        info["reward/base"] = float(reward)

        crashed = bool(info.get("crashed", False))
        if crashed:
            shaped_reward -= self.crash_penalty
            info["reward/crash_penalty"] = -self.crash_penalty
        else:
            shaped_reward += self.survival_bonus
            info["reward/survival_bonus"] = self.survival_bonus

        if self._previous_action is not None and int(action) != self._previous_action:
            shaped_reward -= self.action_switch_penalty
            info["reward/action_switch_penalty"] = -self.action_switch_penalty

        lane_change_delta = self._choose_lane_change_bonus(int(action))
        if lane_change_delta != 0.0:
            shaped_reward += lane_change_delta
            info["reward/lane_change_shaping"] = lane_change_delta

        self._previous_action = int(action)

        info["reward/shaped"] = shaped_reward
        info["safety/crashed"] = crashed
        info["safety/previous_action"] = self._previous_action
        return observation, shaped_reward, terminated, truncated, info
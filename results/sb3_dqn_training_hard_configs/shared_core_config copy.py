SHARED_CORE_ENV_ID = "highway-v0"

SHARED_CORE_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
        "clip": True,
        "see_behind": True,
        "observe_intentions": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [25, 27.5, 30, 32.5], # config 25
    },
    "lanes_count": 4,
    "vehicles_count": 55,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 100,
    "ego_spacing": 2,
    "vehicles_density": 1.2,
    "collision_reward": -10.0,
    "right_lane_reward": 0.02,
    "high_speed_reward": 0.4,
    "lane_change_reward": -0.01,
    "reward_speed_range": [22.5, 30],
    "normalize_reward": True,
    "offroad_terminal": True,
}
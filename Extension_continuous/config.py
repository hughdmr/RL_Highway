# Environment config 

from pathlib import Path
import torch

LOG_DIR         = Path("logs")
SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


ENV_CONFIG = {
    # Observation: same as DQN project
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,
        "absolute": False,
        "see_behind": True,   
    },
    # continuous action space 
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,
        "acceleration_range": [-3.0, 3.0],
        "steering_range": [-0.4, 0.4],
        "speed_range":[25,50] # speed range
    },
    
    "reward_speed_range": [30, 50],
    "collision_reward": -1.5,
    "high_speed_reward": 0.7,

    # Simulation 
    "duration": 100,
    "lanes_count": 3,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "controlled_vehicles": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "offroad_terminal": True,

    "vehicles_count": 15,
    
    "vehicles_density": 0.5,

    "spawn_probability": 0.6,
    "initial_lane_id": None,
}

VISU_CONFIG = {
    # Observation: same as DQN project
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,
        "absolute": False,
        "see_behind": True,   
    },
    # continuous action space 
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,
        "acceleration_range": [-3.0, 3.0],
        "steering_range": [-0.4, 0.4],
        "speed_range":[25,50] # speed range
    },
    
    "reward_speed_range": [33, 50],
    "collision_reward": -1.5,
    "high_speed_reward": 0.7,

    # Simulation 
    "duration": 60,
    "lanes_count": 3,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "controlled_vehicles": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "offroad_terminal": True,

    "vehicles_count": 15,
    
    "vehicles_density": 0.5,

    "spawn_probability": 0.6,
    "initial_lane_id": None,
}

SAC_KWARGS = dict(
    policy            = "MlpPolicy",
    policy_kwargs     = dict(net_arch=[256, 256]),  
    learning_rate     = 1e-3,
    buffer_size       = 100_000,                    
    batch_size        = 128,                        
    gamma             = 0.99,                       
    learning_starts   = 5_000,                      
    train_freq        = 1,
    gradient_steps    = 1,
    verbose           = 1,
    tensorboard_log   = str(LOG_DIR),
    device            = DEVICE,
    seed              = SEED,
    ent_coef          = "auto",   
    target_entropy    = "auto",
    tau               = 0.005,    
    target_update_interval = 1,
)


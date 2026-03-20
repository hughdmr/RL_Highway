# RL_Highway

## From-Scratch DQN (highway-v0)

This repository now includes a custom DQN implementation for the shared benchmark:

- Environment: `highway-v0`
- Configuration: `shared_core_config.py`
- Observation: `Kinematics`
- Action space: `DiscreteMetaAction`

### Files

- `dqn_agent.py`: Q-network, replay buffer, DQN agent logic
- `train_dqn.py`: training loop, checkpoint saving, metrics export
- `evaluate_dqn.py`: evaluation over N episodes (default 50), returns mean/std reward

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python train_dqn.py --run-name dqn_scratch --total-timesteps 200000
```

Outputs are saved under `results/<run-name>/`:

- `checkpoints/best_model.pt`
- `checkpoints/last_model.pt`
- `metrics.csv`
- `summary.json`

### Evaluate (50 runs mean + std)

```bash
python evaluate_dqn.py \
	--checkpoint results/dqn_scratch/checkpoints/best_model.pt \
	--episodes 50 \
	--output-json results/dqn_scratch/eval_50_runs.json
```
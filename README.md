# n-puzzle-RL

A simple n*m tiles environment where an agent move blank tile across the board to swap tiles with numbers in order to achieve the goal state. 
The objective is to reach the state where all tiles are ordered.

<kbd>![15-puzzle](http://mathworld.wolfram.com/images/eps-gif/15Puzzle_1000.gif)</kbd>


### Action space
The agent may only choose to go up, down, left, or right (2, 0, 3, 1). If the way is blocked, it will remain at the same the location. 

### Observation space
The observation space is n*m matrix of integers from 0 (blank tile) to nm - 1.

### Reward
A reward of 100 is given when the agent reaches the goal. For every step, the agent recieves a reward of (current_manhattan_heuristic - previous_manhattan_heuristic), where manhattan_heuristic is the sum of distances between current positions of tiles and positions of goal state.

### End condition
The board is reset when the agent reaches the goal or number of steps >= max_steps. 

## Installation

```bash
cd n-puzzle-RL
python setup.py install
```

# Example Use


### Train agent and run it in sem.ipynb

Configure these params and run

board_sizes = (3, 3)

diff = 100 # difficulty

st = 1000 # max_step

num_of_iter = 100


### Tune hyperparameters in test_env.py

### Another way of usage:

```bash
python ./train.py -f ./puzzle-ppo.yaml
```

```bash
python ./rollout.py ~/ray/checkpoint_dir/checkpoint-0 --run PPO --env puzzle-v0 --steps 1000000 --out rollouts.pkl
```

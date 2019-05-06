**Solving "n-puzzle" game using Reinforcement learning algorithms**

Installation: ``pip install gym_puzzle``

Example Use
-----------

+------------------------------------------------+
| **Train agent and run it in sem.ipynb**        |
+------------------------------------------------+
|Configure these params and run                  |
|.. code-block:: python                          |
|                                                |
|  board_sizes = (3, 3)                          |
|  diff = 100                                    |
|  st = 1000                                     |
|  num_of_iter = 100                             |
+------------------------------------------------+

**Tune hyperparameters in test_env.py**

Another ways of usage:

python ./train.py -f ./puzzle-ppo.yaml

python ./rollout.py ~/ray/checkpoint_dir/checkpoint-0 --run PPO
    --env puzzle-v0 --steps 1000000 --out rollouts.pkl

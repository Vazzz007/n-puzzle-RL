# core modules
import unittest

# 3rd party modules
import gym
from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog
import ray
from ray import tune
from ray.tune import grid_search
from ray.tune.registry import register_env

# internal modules
from gym_puzzle.envs import PuzzleEnv


class CustomModel(Model):

    def _build_layers_v2(self, input_dict, num_outputs, options):
        self.obs_in = input_dict["obs"]
        self.fcnet = FullyConnectedNetwork(input_dict, self.obs_space,
                                           self.action_space, num_outputs,
                                           options)
        return self.fcnet.outputs, self.fcnet.last_layer


if __name__ == "__main__":

    board_sizes = (3, 3)
    diff = 2
    st = 100000000

    env_name = 'puzzle-v0'
    #my_board = gym.make('gym_puzzle:puzzle-v0')
    register_env(env_name, lambda config: PuzzleEnv(config))

    ray.init()
    # ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
        "PPO",
        stop={
            #"timesteps_total": 10000,
            #"episode_len_mean": 20.0,
            "training_iteration": 50,
        },
        config={
            "env": "puzzle-v0",  # or "puzzle-v0" if registered above
            # "model": 
            #     "custom_model": "my_model",
            # },
            "num_gpus": 1,
            #"lr": 1e-5,  # try different lrs
            "num_workers": 6,  # parallelism
            "env_config": {
                "board_sizes": board_sizes,
                "difficulty": diff,
                "max_steps": st,
            },
        },
        resources_per_trial={
         #"cpu": 6,
         #"gpu": 1
        },
    )
import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import numpy as np

from gym.spaces.space import Space

class PuzzleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config = None):

        self.done = False
        self.prev_reward = 0.0
        self.steps_count = 0
        if config == None:
            config = {"max_steps": None, "board_sizes": None, "difficulty": None}
        self.max_steps = config["max_steps"] if config["max_steps"] is not None else 1000000
        self.add = [0, 0]
        self.nrow, self.ncol = config["board_sizes"] if config["board_sizes"] is not None else (2, 3)
        self.board_size = self.nrow * self.ncol
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            0, self.board_size - 1, shape=(self.board_size, ), dtype=np.int32)
        self.board = None
        self.blank_position = None
        self.diff = config["difficulty"] if config["difficulty"] is not None else 10
        self.seed()


    def __repr__(self):
        return "<Board {}>".format(self.board)

    def __eq__(self, other):
        return (self.board == other.board).all()

    def step(self, direction):
        """
        1 = right, 2 = up, 3 = left, 0 = down
        """
        assert self.action_space.contains(direction), "%r (%s) invalid"%(direction, type(direction))

        self.board = self.board.reshape(self.nrow, self.ncol)

        if self.done == True:
            print("Game Over")
            raise RuntimeError("Episode is done")
            return self.board.flatten(), self.prev_reward, self.done, {}
        else:
            x, y = self.blank_position # x is the row, y is the column of the blank tile

            if direction == 1:
                # moving right
                if direction in self.legal_moves():
                    self.board[x][y] = self.board[x][y + 1]
                    self.board[x][y + 1] = 0
                    self.blank_position[1] += 1
    
            elif direction == 2:
                # moving up
                if direction in self.legal_moves():
                    self.board[x][y] = self.board[x - 1][y]
                    self.board[x - 1][y] = 0
                    self.blank_position[0] -= 1
    
            elif direction == 3:
                # moving left
                if direction in self.legal_moves():
                    self.board[x][y] = self.board[x][y - 1]
                    self.board[x][y - 1] = 0
                    self.blank_position[1] -= 1
    
            elif direction == 0:
                # moving down
                if direction in self.legal_moves():
                    self.board[x][y] = self.board[x + 1][y]
                    self.board[x + 1][y] = 0
                    self.blank_position[0] += 1

        #print(self.board, direction, self.reward)



        solve = self.is_solved()
        self.done = False
        self.steps_count += 1
        reward = 0.0

        next_dist = self.manhattan_heuristic()
        next_fits = self.n_wrong_heuristic()

        # if (self.cur_dist >= next_dist):
        #     reward = 100*(1/self.steps_count)*(self.cur_dist - next_dist)
        # else:
        #     reward = 0.001*(self.steps_count)*(self.cur_dist - next_dist)

        reward = 1*(self.cur_dist - next_dist)
                #- 0.01*self.steps_count #+ 10*(next_fits - self.cur_fits)
        self.cur_dist = next_dist
        self.cur_fits = next_fits

        if (solve == True):
            self.done = True
            reward = 100.0

        if (self.steps_count >= self.max_steps):
            self.done = True
        
        self.prev_reward = reward

        return self.board.flatten(), reward, self.done, {"dist": next_dist} #self.add

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def scramble(self):

        def inverse(x):
            return ((x + 6) % 4)

        last_inverse = 0
        for i in range(self.diff):
            step = self.np_random.choice([action for action in self.legal_moves() if action != last_inverse])
            self.move(step)
            last_inverse = inverse(step)

    def get_board(self):
        return self.board

    def render(self):
        print(self.board)

    def reset(self):

        self.board = np.array([x + 1 for x in np.arange(self.board_size)]).astype(np.int32).reshape(self.nrow, self.ncol)
        self.board[self.nrow - 1, self.ncol - 1] = 0
        self.blank_position = [self.nrow - 1, self.ncol - 1]
        self.scramble()
        self.blank_position = np.argwhere(self.board == 0)[0]

        self.prev_reward = 0.0
        self.steps_count = 0
        self.cur_fits = self.n_wrong_heuristic()
        self.done = False
        self.cur_dist = self.manhattan_heuristic()

        return self.board.flatten()


    def is_solved(self):
        solution = self.solved_state()

        if (self.board == solution).all():
            return True
        return False

    def legal_moves(self):
        moves = []
        if self.blank_position[1] != self.ncol - 1:
            moves.append(1)
        if self.blank_position[0] != 0:
            moves.append(2)
        if self.blank_position[1] != 0:
            moves.append(3)
        if self.blank_position[0] != self.nrow - 1:
            moves.append(0)

        return moves

    def solved_state(self):
        a = np.array([x + 1 for x in np.arange(self.board_size)]).astype(np.int32).reshape(self.nrow, self.ncol)
        a[self.nrow - 1, self.ncol - 1] = 0

        return a

    def n_wrong_heuristic(self):
    
        state = self.get_board()
        indices = np.array([np.argwhere(state == i)[0] for i in range(1, self.board_size)])
        correct_indices = np.array([[i, j] for i in range(self.nrow) for j in range(self.ncol)])[:-1]
        n_wrong = 0
        for i,pair in enumerate(indices):
            if (pair != correct_indices[i]).any():
                n_wrong += 1
    
        return n_wrong  
    
    def manhattan_heuristic(self):

        state = self.get_board()
        indices = np.array([np.argwhere(state == i)[0] for i in range(1,self.board_size)])
        correct_indices = np.array([[i, j] for i in range(self.nrow) for j in range(self.ncol)])[:-1]
    
        return np.abs(indices - correct_indices).sum()

    def forecast(self, action):

        new_board = Board(np.copy(self.board))
        new_board.step(action)
        return new_board

    def close(self):
        pass
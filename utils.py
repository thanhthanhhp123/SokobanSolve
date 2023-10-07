import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Sokoban-v0')
env.seed(0)


def calculate_num_states(map, num_actions):
  """Calculates the number of states in a Sokoban environment.

  Args:
    map: A 2D numpy array representing the Sokoban map.

  Returns:
    The number of states in the environment.
  """
  height, width = map.shape
  num_agent_positions = np.count_nonzero(map == 5)
  num_target_cells = np.count_nonzero(map == 2)
  num_box_positions = np.count_nonzero(map == 4)
  num_empty_cell_positions = np.count_nonzero(map == 1)

  num_states_per_agent_position = num_actions ** num_box_positions

  num_states_for_all_agent_positions = num_states_per_agent_position ** num_agent_positions

  num_states_for_all_empty_cell_positions = 2 ** num_empty_cell_positions

  num_states = num_states_for_all_agent_positions * num_states_for_all_empty_cell_positions

  return num_states
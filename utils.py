import gym
import numpy as np
import matplotlib.pyplot as plt

def create_P_function(env):
    P = {}
    for state in env.get_availabel_states:
        for next_states, action in env.get_next_state(state):
            num_next_states = len(next_states)
            prob = [1/ num_next_states] * num_next_states
            P[(state, action)] = list(zip(next_states, prob))
    return P

def calculate_number_states(env):
    num_states = 0
    for i in env.room_state:
        if i != 0:
            num_states += 1
    return num_states
def learnModel(env, num_states, num_action, samples = 1e5):
    reward = np.zeros((num_states, num_action, num_states))
    counter_map = np.zeros((num_states, num_action, num_states))

    counter = 0
    while counter < samples:
        state = env.reset()
        done = False
        
        while True:
            random_action = env.action_space.sample()
            next_state, reward_, done, _ = env.step(random_action)
            counter += 1
            reward[state][random_action][next_state] += reward_

            state = next_state
            counter += 1
            
    counter_map[counter_map == 0] = 1
    reward /= counter_map
    return reward

def plot_evaluation(success_rate, title):
    plt.figure()
    plt.plot(success_rate)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.show()
    plt.savefig(title + '.png', dpi = 300)
    

def testPolicy(env, policy, trials = 100):
    env.reset()
    success = 0

    for _ in range(trials):
        done = False
        state = env.reset()
        while not done:
            action = policy[state]
            state, _, done, _ = env.step(action)
            if state == 0:
                pass
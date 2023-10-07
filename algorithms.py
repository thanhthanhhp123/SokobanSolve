from utils import *
import tqdm


class QLearning():
    def __init__(self, env):
        self.env = env
    def _init_Qtable(self, n_states, n_actions):
        """Initializes the Q-table.
        
        Args:
            n_states: The number of states in the environment.
            n_actions: The number of actions in the environment.
        
        Returns:
            A numpy array of size [n_states, n_actions] containing all zeros.
        """
        return np.zeros((n_states, n_actions))
    def _greedy_policy(self, Qtable, state):
        action = np.argmax(Qtable[state][:])
        return action
    def _epsilon_greedy_policy(self, Qtable, state, epsilon):
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = self.greedy_policy(Qtable, state)
        return action
    
    def train(self, n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, lr, gamma):
        Qtable = self._init_Qtable(n_states=calculate_num_states(env.room_state), n_actions=env.action_space.n)
        for episode in tqdm(range(n_training_episodes)):
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            state = self.env.reset()
            done = False
            step = 0

            for step in range(max_steps):
                action = self.epsilon_greedy_policy(Qtable, state, epsilon)

                new_state, reward, done, info = env.step(action)

                Qtable[state][action] = Qtable[state][action] + lr * (reward + gamma * np.max(Qtable[new_state][:]) - Qtable[state][action])
                if done:
                    break
                state = new_state
        return Qtable

if __name__ == '__main__':
    QLearning = QLearning(env)
    print(QLearning._init_Qtable)
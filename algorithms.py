from utils import *
import tqdm


class PolicyIteration():
    def __init__(self, env, V, S, A, P):
        self.env = env
        self.V = V
        self.S = S
        self.A = A
        self.P = P
    
    def _init_Vtable(self, n_states):
        """Initializes the V-table.
        
        Args:
            n_states: The number of states in the environment.
        
        Returns:
            A numpy array of size [n_states] containing all zeros.
        """
        return np.zeros(n_states)
    def policy_improvement(self, policy, gamma):
        policy_stable = True
        num_state = calculate_number_states(self.env.room_state)
        num_action = self.env.action_space.n

        for s in range(num_state):
            old_action = policy[s]
            val = self.V[s]
            for a in range(num_action):
                tmp = 0
                for s_new in range(num_state):
                    tmp += self.P[s][a][s_new] * (
                        self.R[s][a][s_new] + gamma * self.V[s_new]
                        )
                if tmp > val:
                    policy[s] = a
                    val = tmp
            if old_action != policy[s]:
                policy_stable = False
        return policy, policy_stable
    def policyEval(self, policy, gamma, max_iteration = 1000):
        counter = 0
        num_state = calculate_number_states(self.env)
        
        while counter < max_iteration:
            counter += 1
            for s in range(num_state):
                val = 0
                for s_new in range(num_state):
                    val += self.P[s][policy[s]][s_new] * (
                        self.R[s][policy[s]][s_new] + gamma * self.V[s_new]
                        )
                self.V[s] = val
        return self.V
    
    def train(self, gamma = 0.99, max_iteration = 1000, stop_if_stable = False):
        success_rate = []
        num_state = calculate_number_states(self.env)
        
        policy = np.zeros(num_state, dtype = int)
        value = self._init_Vtable(num_state)

        counter = 0
        while counter < max_iteration:
            counter += 1
            value = self.policyEval(policy, gamma)
            policy, policy_stable = self.policy_improvement(policy, gamma)
            # success_rate.append(self.evaluate(policy, gamma))
            if policy_stable and stop_if_stable:
                print("policy is stable at {} iteration".format(counter))
        return policy

        
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
            action = self.env.action_space.sample()
        else:
            action = self.greedy_policy(Qtable, state)
        return action
    
    def train(self, n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, lr, gamma):
        Qtable = self._init_Qtable(n_states=calculate_number_states(env.room_state), n_actions=env.action_space.n)
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

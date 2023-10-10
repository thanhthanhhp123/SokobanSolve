from utils import *
import tqdm

'''
Params:
    gamma: discount factor, value of (0, 1), represent how much future value func affect
'''

class PolicyIteration():
    # Using iterative policy evaluation

    def __init__(self, env, V, S, P, R):
        self.env = env
        self.V = V
        self.S = S
        self.P = P
        self.R = R
    
    def _init_Vtable(self, n_states):
        # Initialize V table as a zero values numpy array of size [number of states]
        return np.zeros(n_states)
    
    def policy_improvement(self, policy, gamma):
        policy_stable = True
        num_state = calculate_number_states(self.env.room_state)
        num_action = self.env.action_space.n

        for s in range(num_state):
            old_action = policy[s]
            val = self.V[s]
            for a in range(num_action):
                tmp = 0     # stores updated V[each state] by all actions
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
        
        # Update value function via Bellman equation
        while counter < max_iteration:
            counter += 1
            for s in range(num_state):
                val = 0     # stores updated V[specified action] by all states
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
                break
        return policy


class ValueIteration():
    def __init__(self, env, P, R):
        self.env = env
        self.P = P 
        self.R = R

    def _init_V(self, n_states):
        # Initialize V table as a zero values numpy array of size [number of states]
        return np.zeros(n_states)

    def value_improvement(self, state, gamma, V):
        # Update V via Bellman equation in a specified state
        n_states = calculate_number_states(self.env)
        n_actions = self.env.action_space.n
        s = state

        tmp = np.zeros(n_actions) # array stores all updated V[specified state] by all actions

        for a in range(n_actions):
            for s_new in range(n_states):
                tmp[a] += self.P[s][a][s_new] * (self.R[s][a][s_new] + gamma * V[s_new])

        V[s] = np.max(tmp)
        choose_action = np.argmax(tmp)

        return V[s], choose_action
    
    def train(self, gamma, max_iteration):
        n_states = calculate_number_states(self.env)
        Vtable = self._init_V(n_states)
        policy = np.zeros(n_states, dtype = int)

        for _ in range(max_iteration):
            for s in range(n_states):
                Vtable[s], policy[s] = self.value_improvement(state=s, gamma=gamma, V=Vtable)

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
            action = self._greedy_policy(Qtable, state)
        return action
    
    def train(self, n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, lr, gamma):
        Qtable = self._init_Qtable(n_states=calculate_number_states(env.room_state), n_actions=env.action_space.n)
        for episode in tqdm(range(n_training_episodes)):
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            state = self.env.reset()
            state = 10 * state[0] + state[1]
            done = False

            for _ in range(max_steps):
                action = self._epsilon_greedy_policy(Qtable, state, epsilon)

                new_state, reward, done, info = env.step(action)

                Qtable[state][action] = Qtable[state][action] + lr * (reward + gamma * np.max(Qtable[new_state][:]) - Qtable[state][action])
                if done:
                    break
                state = new_state
        return Qtable
    

class SARSA():
    def __init__(self, env):
        self.env = env 

    def _init_Qtable(self, n_states, n_actions):
        # Initialize Q table
        return np.zeros((n_states, n_actions))
    
    def _greedy_policy(self, Qtable, state):
        action = np.argmax(Qtable[state][:])
        return action
    
    def _epsilon_greedy_policy(self, Qtable, state, epsilon):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self._greedy_policy(Qtable, state)
        return action
    
    def train(self, episodes, min_epsilon, max_epsilon, decay_rate, max_steps, lr, gamma):
        n_states = calculate_number_states(self.env)
        n_actions = self.env.action_space.n
        Qtable = self._init_Qtable(n_states, n_actions)

        for episode in tqdm(range(episodes)):
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            state = self.env.reset()
            s = 10 * state[0] + state[1]
            done = False

            action = self._epsilon_greedy_policy(Qtable, s, epsilon)
            for _ in range(max_steps):
                new_state, reward, done, info = self.env.step(action)
                s_new = 10 * new_state[0] + new_state[1]

                a_new = self._epsilon_greedy_policy(Qtable, s_new, epsilon)

                Qtable[s][action] = Qtable[s][action] + lr * (reward + gamma * Qtable[s_new][a_new] - Qtable[s][action])
                if done:
                    break
                s = s_new
                action = a_new

        return Qtable
    

class MonteCarlo():
    # Using first-vist Monte Carlo prediction
    def __init__(self, env):
        self.env = env 

    def _init_V(self, n_states):
        # Initialize V table as a zero values numpy array of size [number of states]
        return np.zeros(n_states)
    
    def _sample_policy(self):
        action = self.env.action_space.sample()
        return action
    
    def generate_episode(self, max_steps):
        # we initialize the list for storing states, actions, and rewards
        states, actions, rewards = [], [], []
        
        # Initialize the gym environment
        observation = self.env.reset()
        
        for _ in range(max_steps):    
            # append the states to the states list
            states.append(10 * observation[0] + observation[1])
            
            # now, we select an action using our sample_policy function and append the action to actions list           
            action = self._sample_policy()
            actions.append(action)
            
            # We perform the action in the environment according to our sample_policy, move to the next state 
            # and receive reward
            observation, reward, done, info = self.env.step(action)
            rewards.append(reward)
            
            # Break if the state is a terminal state
            if done:
                break                    
        return states, actions, rewards
    
    def train(self, n_episodes, max_steps):    
        # First, we initialize the empty value table as a dictionary for storing the values of each state
        n_states = calculate_number_states(self.env)
        Vtable = self._init_V(n_states)
        N = np.zeros(n_states, dtype = int)

        
        for _ in range(n_episodes):
            
            # Next, we generate the epsiode and store the states and rewards
            states, actions, rewards = self.generate_episode(max_steps)
            returns = 0
            
            # Then for each step, we store the rewards to a variable R and states to S, and we calculate
            # returns as a sum of rewards
            
            for t in range(len(states) - 1, -1, -1):
                R = rewards[t]
                S = states[t]
                
                returns += R
                
                # Now to perform first visit MC, we check if the episode is visited for the first time, if yes,
                # we simply take the average of returns and assign the value of the state as an average of returns
                
                if S not in states[:t]:
                    N[S] += 1
                    Vtable[S] += (returns - Vtable[S]) / N[S]        
        return Vtable
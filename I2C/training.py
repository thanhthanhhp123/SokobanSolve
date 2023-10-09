import  numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim

from common.utils import hwc2chw
from common.actor_critic import ActorCritic, RolloutStorage
from common.logger import Logger
from common.multiprocess_env import SubprocVecEnv
from common.timer import Timer
from tqdm import tqdm
import gym
import warnings
warnings.filterwarnings("ignore")

logger = Logger()
timer = Timer()

USE_CUDA = torch.cuda.is_available()

num_envs = 16
env_name = "Boxoban-Train-v0"

def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env
    return _thunk

if __name__ == '__main__':
    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    state_shape = envs.observation_space.shape
    num_actions = envs.action_space.n
    gamma = 0.99
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    num_steps = 5
    num_frames = int(10e6)

    lr = 7e-4
    eps = 1e-5
    alpha = 0.99

    actor_critic = ActorCritic(state_shape, num_actions)
    optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)
    if USE_CUDA:
        actor_critic = actor_critic.cuda()
    
    rollout = RolloutStorage(num_steps, num_envs, state_shape)
    if USE_CUDA:
        rollout.cuda()
    
    all_rewards = []
    all_losses = []
    all_step_scores = []

    state = envs.reset()

    state = torch.FloatTensor(np.float32(state))
    if USE_CUDA:
        state = state.cuda()
    
    rollout.states[0].copy_(state)
    episode_rewards = torch.zeros(num_envs, 1)
    final_rewards = torch.zeros(num_envs, 1)
    timer.update(time.time())
    print("Start training")
    for i_update in tqdm(range(num_frames)):
        for step in range(num_steps):
            action = actor_critic.act(state)

            next_state, reward, done, info = envs.step(action.squeeze(1).cpu().data.numpy())
            episode_rewards += reward
            # finished_masks = torch.FloatTensor(1-np.array(finished)).unsqueeze(1)

            # final_rewards *= finished_masks
            # final_rewards += (1-finished_masks) * episode_rewards

            # episode_rewards *= finished_masks
            state = torch.FloatTensor(np.float32(next_state))

            if USE_CUDA:
                # finished_masks = finished_masks.cuda()
                state = state.cuda()
            rollout.insert(step, state, action.data, torch.tensor(reward))
        
        _, next_value = actor_critic(rollout.states[-1])
        next_value = next_value.data

        returns = rollout.compute_returns(next_value, gamma)

        logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(
            rollout.states[:-1].view(-1, *state_shape),
            rollout.actions.view(-1, 1)
        )

        values = values.view(num_steps, num_envs, 1)
        action_log_probs = action_log_probs.view(num_steps, num_envs, 1)
        advantages = returns - values

        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.data * action_log_probs).mean()

        optimizer.zero_grad()
        loss = value_loss * value_loss_coef + action_loss - entropy * entropy_coef
        loss.backward()
        nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
        optimizer.step()

        if i_update % 100 == 0:
            all_rewards.append(final_rewards.mean())
            all_losses.append(loss.item())
            timer.update(time.time())
            loopstogo = (num_frames - i_update) / 100
            estimatedtimetogo = timer.getTimeToGo(loopstogo)

            print('Epoch %s Reward %s' % (i_update, torch.tensor(all_rewards[-10:]).mean()))
            print('Loss %s' % all_losses[-1])
            logger.printDayFormat('Estimated time to run:', seconds= estimatedtimetogo)
        
        rollout.after_update()
    
    logger.log(all_rewards, "Data/", "all_rewards.txt")
    logger.log(all_losses, "Data/", "all_losses.txt")
    logger.log_state_dict(actor_critic.state_dict(), "Data/actor_critic")
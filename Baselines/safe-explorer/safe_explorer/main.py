import time
from datetime import datetime
from functional import seq
import numpy as np
import torch

from safe_explorer.core.config import Config
from safe_explorer.core.tensorboard import TensorBoard
from safe_explorer.env.ballnd import BallND
from safe_explorer.env.spaceship import Spaceship
from safe_explorer.safety_layer.safety_layer import SafetyLayer

from safe_explorer.ddpg.ddpg import DDPGAgent
from safe_explorer.ddpg.utils import OUNoise


class Trainer:
    def __init__(self):
        config = Config.get().main.trainer
        # set seeds
        torch.manual_seed(config.seed)
        np.random.seed(int(config.seed))

        self.use_safety_layer = config.use_safety_layer

        # create environment
        if config.task == 'ballnd':
            self.env = BallND()
        else:
            self.env = Spaceship()

    def train(self):
        print("============================================================")
        print("Initialized SafeExplorer with config:")
        print("------------------------------------------------------------")
        Config.get().pprint()
        print("============================================================")

        # init Safety Layer
        safety_layer = None
        if self.use_safety_layer:
            safety_layer = SafetyLayer(self.env)
            safety_layer.train()
        # obtain state and action dimensions
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # get config
        config = Config.get().ddpg.trainer
        # get relevant config values
        epochs = config.epochs
        training_episodes = config.training_episodes_per_epoch
        evaluation_episodes = config.evaluation_episodes_per_epoch
        # max_episode_length = config.max_episode_length
        batch_size = config.batch_size

        # create agent
        agent = DDPGAgent(state_dim, action_dim)
        # create exploration noise
        noise = OUNoise(self.env.action_space)
        # metrics for tensorboard
        cum_constr_viol = 0  # cumulative constraint violations
        eval_step = 0
        # create Tensorboard writer
        writer = TensorBoard.get_writer()

        start_time = time.time()
        print("==========================================================")
        print("Initializing DDPG training...")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        for epoch in range(epochs):
            # training phase
            agent.set_train_mode()
            for _ in range(training_episodes):
                noise.reset()
                state = self.env.reset()
                done = False
                while not done:
                    # get original policy action
                    action = agent.get_action(state)
                    # add OU-noise
                    action = noise.get_action(action)
                    # get safe action
                    if safety_layer:
                        constraints = self.env.get_constraint_values()
                        action = safety_layer.get_safe_action(
                            state, action, constraints)
                    # apply action
                    next_state, reward, done, _ = self.env.step(action)
                    # push to memory
                    agent.memory.push(state, action, reward, next_state, done)
                    # update agent
                    if len(agent.memory) > batch_size:
                        agent.update(batch_size)
                    state = next_state
            print(f"Finished epoch {epoch}. Running evaluation ...")

            # evaluation phase
            agent.set_eval_mode()
            episode_rewards, episode_lengths, episode_actions = [], [], []
            for _ in range(evaluation_episodes):
                state = self.env.reset()
                episode_action, episode_reward, episode_step = 0, 0, 0
                done = False
                # render environment
                # self.env.render()
                while not done:
                    # get original policy action
                    action = agent.get_action(state)
                    # get safe action
                    if safety_layer:
                        constraints = self.env.get_constraint_values()
                        action = safety_layer.get_safe_action(
                            state, action, constraints)
                    episode_action += np.absolute(action)
                    # apply action
                    state, reward, done, info = self.env.step(action)
                    episode_step += 1
                    # update metrics
                    episode_reward += reward
                    # render environment
                    # self.env.render()
                    
                if 'constraint_violation' in info and info['constraint_violation']:
                    cum_constr_viol += 1
                # log metrics to tensorboard
                writer.add_scalar("metrics/episode length",
                                  episode_step, eval_step)
                writer.add_scalar("metrics/episode reward",
                                  episode_reward, eval_step)
                writer.add_scalar("metrics/cumulative constraint violations",
                                  cum_constr_viol, eval_step)
                eval_step += 1
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_step)
                episode_actions.append(episode_action / episode_step)

            print("Evaluation completed:\n"
                  f"Number of episodes: {len(episode_actions)}\n"
                  f"Average episode length: {np.mean(episode_lengths)}\n"
                  f"Average reward: {np.mean(episode_rewards)}\n"
                  f"Average action magnitude: {np.mean(episode_actions)}\n"
                  f"Cumulative Constraint Violations: {cum_constr_viol}")
            print("----------------------------------------------------------")
        print("==========================================================")
        print(
            f"Finished DDPG training. Time spent: {(time.time() - start_time) // 1} secs")
        print("==========================================================")
        # close environment
        self.env.close()
        # close tensorboard writer
        writer.close()


if __name__ == '__main__':
    Trainer().train()

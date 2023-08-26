import os
import gym
import numpy as np
import argparse
from utils import create_directory, plot_learning_curve
from D3QN import D3QN
import malware_rl
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=1100)
parser.add_argument('--per', type=bool, default=True) # use prioritized experience replay instead random
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/EMBER_PSP/') # save path of model
parser.add_argument('--reward_path', type=str, default='./output_images/reward.png') # log reward curve
parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon.png') # log epsilon curve

args = parser.parse_args()

# define a random agent
class RandomAgent:
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, observation):
        return self.action_space.sample()

def main():
    env = gym.make('ember-train-v0')

    # define D3QN agent
    agent = D3QN(alpha=0.0003, state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                 fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.99, tau=0.005, epsilon=1,
                 eps_end=0.1, eps_dec=5e-4, max_size=500000, batch_size=256, per=args.per,)


    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    total_rewards, avg_rewards, epsilon_history = [], [], []

    for episode in range(args.max_episodes):
        total_reward = 0
        done = False
        observation, p_shapley = env.reset()
        while not done:
            p_shapley = env.p_shapley
            action = agent.choose_action(observation, p_shapley, isTrain=True)
            observation_, reward, done, info = env.step(action)

            if args.per != True:
                agent.remember(observation, action, reward, observation_, done) # without per
            else:
                agent.remember_shapley(observation, action, reward, observation_, done, p_shapley) # adopt per

            agent.learn()
            total_reward += reward
            observation = observation_

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-50:])
        avg_rewards.append(avg_reward)
        epsilon_history.append(agent.epsilon)
        print('EP:{} Reward:{} Avg_reward:{} Epsilon:{}'.
              format(episode+1, total_reward, avg_reward, agent.epsilon))

        # save model and thompson sampling weight
        if (episode + 1) % 100 == 0:
            agent.save_models(episode+1)
            with open(args.ckpt_dir + 'arm.pkl', 'wb') as file:
                pickle.dump(env.arms, file, True)

        reward_file = open('ember_psp.txt', 'a')
        reward_file.write(str(avg_reward) + '\n')
        reward_file.close()



    print('---------- train over ------------')
    print('---------- start test ------------')

    env_test = gym.make('sorel-test-v0')

    # agent = RandomAgent(env_test.action_space)

    episode_count = 400

    agent.load_models(1400)

    # load thompson sampling weight
    with open(args.ckpt_dir + 'arm.pkl', 'rb') as file:
        env_test.arms = pickle.loads(file.read())


    max_turn = env_test.maxturns

    evasions = 0
    evasion_history = {}

    for i in range(episode_count):
        total_reward = 0
        done = False
        observation, p_shapley = env_test.reset()
        sha256 = env_test.sha256
        num_turn = 0

        while num_turn < max_turn:
            p_shapley = env_test.p_shapley
            action = agent.choose_action(observation, p_shapley, isTrain=False)
            # action = agent.choose_action(observation) # random agent
            observation_, reward, done, ep_history = env_test.step(action)
            total_reward += reward
            observation = observation_

            num_turn = env_test.turns
            if done and reward >= 10.0:
                evasions += 1
                evasion_history[sha256] = ep_history
                break

            elif done:
                break

    # Output metrics/evaluation stuff
    evasion_rate = (evasions / episode_count) * 100
    print(f"{evasion_rate}% samples evaded model.")

    # write evasion_history to txt file
    file = open('history_sorel.txt', 'w')
    for k, v in evasion_history.items():
        file.write(str(k) + ' ' + str(v) + '\n')
    file.close()



if __name__ == '__main__':
    main()

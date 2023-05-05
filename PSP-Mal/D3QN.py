import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from prioritized_memory import Memory
from torch.autograd import Variable

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.V = nn.Linear(fc2_dim, 1)
        self.A = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))

        V = self.V(x)
        A = self.A(x)
        Q = V + A - T.mean(A, dim=-1, keepdim=True)

        return Q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class D3QN:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, ckpt_dir,
                 gamma=0.99, tau=0.005, epsilon=1.0, eps_end=0.01, eps_dec=5e-7,
                 max_size=1000000, batch_size=256, eps_shapley=False, per=True):
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.action_space = [i for i in range(action_dim)]

        self.q_eval = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                          fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=fc1_dim, fc2_dim=fc2_dim)

        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
                                   max_size=max_size, batch_size=batch_size)

        self.per_memory = Memory(max_size)

        self.update_network_parameters(tau=1.0)

        self.eps_shapley = eps_shapley

        self.per = per

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, state, action, reward, state_, done):

        self.memory.store_transition(state, action, reward, state_, done)


    def remember_shapley(self, state, action, reward, state_, done, p_shapely):


        target = self.q_eval(T.tensor(state, dtype=T.float).to(device)).data
        old_val = target[action]
        target_val = self.q_target(T.tensor(state_, dtype=T.float).to(device)).data

        new_val = reward + self.gamma * T.max(target_val).cpu().detach().numpy()

        error = abs(old_val.cpu().detach().numpy() - new_val)

        pa_shapely = p_shapely[action]

        self.per_memory.add(error, pa_shapely, (state, action, reward, state_, done))

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, observation, p_shapley, isTrain=True):
        state = T.tensor([observation], dtype=T.float).to(device)
        q_vals = self.q_eval.forward(state)
        action = T.argmax(q_vals).item()

        if (np.random.random() < self.epsilon) and isTrain:
                action = np.random.choice(self.action_space)

        return action

    def learn(self):

        if self.per != True:

            if not self.memory.ready():
                return

            states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
            batch_idx = T.arange(self.batch_size, dtype=T.long).to(device)
            states_tensor = T.tensor(states, dtype=T.float).to(device)
            actions_tensor = T.tensor(actions, dtype=T.long).to(device)
            rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
            next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
            terminals_tensor = T.tensor(terminals).to(device)

            with T.no_grad():
                q_ = self.q_target.forward(next_states_tensor)
                max_actions = T.argmax(self.q_eval.forward(next_states_tensor), dim=-1)
                q_[terminals_tensor] = 0.0
                target = rewards_tensor + self.gamma * q_[batch_idx, max_actions]
            q = self.q_eval.forward(states_tensor)[batch_idx, actions_tensor]
            loss = F.mse_loss(q, target.detach())

        else:
            mini_batch, idxs, is_weights = self.per_memory.sample(self.batch_size)
            batch_idx = T.arange(self.batch_size, dtype=T.long).to(device)

            mini_batch = np.array(mini_batch).transpose()

            states = np.vstack(mini_batch[0])
            actions = list(mini_batch[1])
            rewards = list(mini_batch[2])
            next_states = np.vstack(mini_batch[3])
            dones = mini_batch[4]
            dones = dones.astype(int)

            states_tensor = T.tensor(states, dtype=T.float).to(device)
            actions_tensor = T.tensor(actions, dtype=T.long).to(device)
            rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
            next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
            terminals_tensor = T.tensor(dones).to(device)

            with T.no_grad():
                q_ = self.q_target.forward(next_states_tensor)
                max_actions = T.argmax(self.q_eval.forward(next_states_tensor), dim=-1)
                q_[terminals_tensor] = 0.0
                target = rewards_tensor + self.gamma * q_[batch_idx, max_actions]
            q = self.q_eval.forward(states_tensor)[batch_idx, actions_tensor]

            errors = T.abs(q - target.detach()).cpu().detach().numpy()  # compute td error

            # update priority
            for i in range(self.batch_size):
                idx = idxs[i]
                self.per_memory.update(idx, errors[i])

            loss = (T.tensor(is_weights, dtype=T.float).to(device) * F.mse_loss(q, target.detach())).mean()

        # update weight of net

        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.update_network_parameters()
        self.decrement_epsilon()

    def save_models(self, episode):
        self.q_eval.save_checkpoint(self.checkpoint_dir + 'Q_eval/D3QN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')
        self.q_target.save_checkpoint(self.checkpoint_dir + 'Q_target/D3QN_Q_target_{}.pth'.format(episode))
        print('Saving Q_target network successfully!')

    def load_models(self, episode):
        self.q_eval.load_checkpoint(self.checkpoint_dir + 'Q_eval/D3QN_q_eval_{}.pth'.format(episode))
        print('Loading Q_eval network successfully!')
        self.q_target.load_checkpoint(self.checkpoint_dir + 'Q_target/D3QN_Q_target_{}.pth'.format(episode))
        print('Loading Q_target network successfully!')

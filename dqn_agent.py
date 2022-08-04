import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from replay_buffer import ReplayBuffer
import pathlib


class DQNAgent():
    def __init__(self, input_shape, action_size, seed, device, buffer_size, batch_size, gamma, lr, tau, update_every,
                 replay_after, model, load_trained_model, save_every, root_dir, model_name):
        """Initialize an Agent object.

        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            buffer_size (int): replay buffer size
            batch_size (int):  minibatch size
            gamma (float): discount factor
            lr (float): learning rate
            update_every (int): how often to update the network
            replay_after (int): After which replay to be started
            model(Model): Pytorch Model
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every
        self.replay_after = replay_after
        self.DQN = model
        self.tau = tau
        self.root_dir = root_dir

        # Q-Network
        self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        self.target_net = self.DQN(input_shape, action_size).to(self.device)

        if load_trained_model:
            load_model(self.root_dir, model_name[0], self.policy_net)
            load_model(self.root_dir, model_name[1], self.target_net)


        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.seed, self.device)

        self.t_step = 0
        self.save_every = save_every

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_after:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def act_test(self, state, eps=0.):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        # self.policy_net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from policy model
        Q_expected_current = self.policy_net(states)
        Q_expected = Q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_net(next_states).detach().max(1)[0]

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net, self.tau)

    # θ'=θ×τ+θ'×(1−τ)
    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def save_networks(self, curr_episode, fin_episode):
        if curr_episode % self.save_every == 0:
            save_model(self.root_dir, "ep-{}_policy_net".format(str(curr_episode)), self.policy_net)
            save_model(self.root_dir, "ep-{}_target_net".format(str(curr_episode)), self.target_net)
        if curr_episode == fin_episode:
            save_model(self.root_dir, "policy_net_fin", self.policy_net)
            save_model(self.root_dir, "target_net_fin", self.target_net)

def save_model(root, model_name, model):
    filepath = pathlib.Path(root + model_name + '.pt')
    if not filepath.parent.exists():
        filepath.parent.mkdir()
    torch.save(model.state_dict(), str(filepath))
    print(model_name + " saved!")

def load_model(root, model_name, model):
    filepath = pathlib.Path(root + model_name + '.pt')
    print('1')
    if filepath.exists():
    	print('2')
    	model.load_state_dict(torch.load(str(filepath)))
    	print(model_name + " loaded!")

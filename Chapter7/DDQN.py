import torch
import random
import copy
from torch.autograd import Variable
import torch.nn as nn

class DuelingModel(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(DuelingModel, self).__init__()
        self.adv1 = nn.Linear(n_input, n_hidden)
        self.adv2 = nn.Linear(n_hidden, n_output)
        self.val1 = nn.Linear(n_input, n_hidden)
        self.val2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        adv = nn.functional.relu(self.adv1(x))
        adv = self.adv2(adv)
        val = nn.functional.relu(self.val1(x))
        val = self.val2(val)
        return val + adv - adv.mean()


class DQN():
    def __init__(self, n_state, n_action, n_hidden = 50, lr = 0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = DuelingModel(n_state, n_action, n_hidden)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, s, y):
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        with torch.no_grad():
            return self.model(torch.Tensor(s))

    def replay(self, memory, replay_size, gamma):
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            states = []
            td_targets = []
            for state, action, next_state, reward, is_done_1, is_done_2 in replay_data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if is_done_1 or is_done_2:
                    q_values[action] = reward
                else:
                    q_values_next = self.predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                td_targets.append(q_values)
            self.update(states, td_targets)
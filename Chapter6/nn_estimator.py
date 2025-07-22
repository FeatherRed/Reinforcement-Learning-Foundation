import torch
from torch.autograd import Variable
import math


class Estimator():
    def __init__(self, n_feat, n_state, n_action, n_hidden, lr = 0.05):
        self.w, self.b = self.get_gaussian_wb(n_feat, n_state)
        self.n_feat = n_feat
        self.hidden = n_hidden
        self.models = []
        self.optimizers = []
        self.criterion = torch.nn.MSELoss()
        for _ in range(n_action):
            model = torch.nn.Sequential(
                torch.nn.Linear(n_feat, n_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(n_hidden, 1)
            )
            self.models.append(model)
            optimizer = torch.optim.Adam(model.parameters(), lr)
            self.optimizers.append(optimizer)

    def get_gaussian_wb(self, n_feat, n_state, sigma = 0.2):
        torch.manual_seed(0)
        w = torch.randn(n_state, n_feat) * 1.0 / sigma
        b = torch.rand(n_feat) * 2.0 * math.pi
        return w, b

    def get_feature(self, s):
        features = (2.0 / self.n_feat) ** .5 * torch.cos(
            torch.matmul(torch.tensor(s).float(), self.w)
            + self.b
        )
        return features

    def update(self, s, a, y):
        features = Variable(self.get_feature(s))
        y_pred = self.models[a](features)
        loss = self.criterion(y_pred, Variable(torch.Tensor([y])))
        self.optimizers[a].zero_grad()
        loss.backward()
        self.optimizers[a].step()

    def predict(self, s):
        features = self.get_feature(s)
        with torch.no_grad():
            return torch.tensor([model(features)
                                 for model in self.models])

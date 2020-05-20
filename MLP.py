from torch import nn


class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(MLP, self).__init__()

        self.body = nn.Sequential(
            nn.Linear(n_in, n_hidden),

            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),

            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_out)
        )

    def forward(self, x):
        return self.body(x.view(x.size(0), -1))

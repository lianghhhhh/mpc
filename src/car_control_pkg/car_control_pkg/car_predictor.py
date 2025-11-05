# state space model based neural network model for car state prediction
# input: u & x, output: 4 matrix A, B defining the state space model
import torch
import torch.nn as nn

class CarPredictor(nn.Module):
    def __init__(self, u_size=4, x_size=4, hidden_size=128, output_size=32, dropout=0.2):
        super().__init__()
        self.u_size = u_size
        self.x_size = x_size
        self.input_size = x_size + u_size
        self.model = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, input):
        # u = input[:self.u_size]
        # x = input[self.u_size:self.u_size + self.x_size]
        # u_flat = u.view(-1, self.u_size)
        # x_flat = x.view(-1, self.x_size)
        input_flat = input.view(-1, self.input_size)

        outputs = self.model(input_flat)
        outputs = self.layer_norm(outputs)

        # A = outputs[:, :self.x_size * self.x_size].reshape(-1, self.x_size, self.x_size)
        # B = outputs[:, self.x_size * self.x_size:self.x_size * self.x_size + self.x_size * self.u_size].reshape(-1, self.x_size, self.u_size)

        return outputs
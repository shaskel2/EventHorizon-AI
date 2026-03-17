import torch
import torch.nn as nn

class OpticalEventHorizonPINN(nn.Module):
    def __init__(self, num_layers=6, hidden_dim=100):
        super().__init__()
        
        # Input layer: 2 dimensions (x, t) -> hidden_dim
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        
        # Build the hidden layers dynamically
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        # Output layer: 2 dimensions (u, v) representing the complex optical field
        layers.append(nn.Linear(hidden_dim, 2))
        
        # Bundle into a sequential module
        self.network = nn.Sequential(*layers)
        
    def forward(self, x, t):
        # Concatenate space (x) and time (t) into a single (N, 2) input tensor
        inputs = torch.cat([x, t], dim=1)
        
        # Pass through the neural network
        outputs = self.network(inputs)
        
        # Split the output back into real (u) and imaginary (v) components
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        
        return u, v

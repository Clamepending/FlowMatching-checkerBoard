import torch
import math
import torch.nn as nn


class FlowMatchingModel(torch.nn.Module):
    def __init__(self, data_dim, hidden_dim, num_hidden_layers=1):
        super(FlowMatchingModel, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        
        # First layer takes data as input
        self.projection = torch.nn.Linear(data_dim, hidden_dim)
        self.projection_activation = torch.nn.ReLU()
        
        # Time encoding dimension
        self.time_dim = hidden_dim
        
        
        self.time_projection = torch.nn.Linear(hidden_dim, hidden_dim)
        
        # First hidden layer after concatenation takes 2*hidden_dim as input
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.ReLU())
        
        # Remaining hidden layers
        for i in range(num_hidden_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(torch.nn.ReLU())
        
        # Output layer
        self.layers.append(torch.nn.Linear(hidden_dim, data_dim))

    def forward(self, x, t):
        # Project input data
        x = self.projection(x)
        x = self.projection_activation(x)
        
        # Encode time
        t_encoding = self.position_encoding(t)
        
        t_encoding = self.time_projection(t_encoding)
        
        # Concatenate along feature dimension
        x = x + t_encoding
        
        # Pass through network layers
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def position_encoding(self, t):
        """
        Generate position encodings based on input time values.
        
        Args:
            t: Time tensor of shape [batch_size]
            
        Returns:
            Position encoding tensor of shape [batch_size, hidden_dim]
        """
        batch_size = t.shape[0]
        
        # Create position encoding tensor
        position_encoding = torch.zeros(batch_size, self.time_dim, device=t.device)
        
        # Create frequency bands
        half_dim = self.time_dim // 2
        frequencies = torch.exp(
            torch.arange(0, half_dim, device=t.device) * 
            (-math.log(10000.0) / (half_dim - 1))
        )
        
        # Reshape t for broadcasting: [batch_size, 1]
        t_expanded = t.view(batch_size, 1)
        
        # Compute arguments for sin and cos: [batch_size, half_dim]
        args = t_expanded * frequencies
        
        # Fill position encoding with sin and cos values
        position_encoding[:, 0::2] = torch.sin(args)
        position_encoding[:, 1::2] = torch.cos(args)
        
        return position_encoding
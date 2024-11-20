import torch.nn as nn
import torch

class FullyConnectedNN(nn.Module):
    """
    A fully connected neural network with a variable number of hidden layers.
    
    Args:
    input_size: int, number of input features
    hidden_size: int, number of hidden units
    num_classes: int, number of output classes
    num_hidden_layers: int, number of hidden layers
    activation: torch activation function, the activation function to use
    
    """
    def __init__(self, input_size, hidden_size, num_hidden_layers, activation, dropout_rate, decay_factor, num_classes=20):
        super(FullyConnectedNN, self).__init__()
        # Define the layers
        layers = []

        # Input layer
        current_size = hidden_size
        layers.append(nn.Linear(input_size, current_size))
        layers.append(activation)
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(num_hidden_layers-1):
            next_size = max(1, int(current_size*decay_factor))
            layers.append(nn.Linear(current_size, next_size))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            current_size = next_size

        # Output layer
        layers.append(nn.Linear(current_size, num_classes))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        # Define the forward pass
        x = x.view(x.size(0), -1)
        return self.layers(x)
    

    
# Instantiate the model
# model = FullyConnectedNN(input_size=1568, hidden_size=128, num_classes=20, num_hidden_layers=2, activation=nn.ReLU())

# # Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Define the data loaders
# train_data = TensorDataset(train_images_tensor, train_labels_tensor)
# val_data = TensorDataset(val_images_tensor, val_labels_tensor)
# test_data = TensorDataset(test_images_tensor, test_labels_tensor)

# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

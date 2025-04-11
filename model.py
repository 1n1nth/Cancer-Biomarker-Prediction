import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneSelectorNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim1=128, hidden_dim2=64, dropout=0.3):
        super(GeneSelectorNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)  # Outputs mutation probabilities
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) # Prevent overfitting

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)  # Output shape: (samples, num_classes)
        x = torch.sigmoid(x)  # Mutation probability scores
        return x

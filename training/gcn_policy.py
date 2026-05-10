
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCNFeatureExtractor(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=64, output_dim=128):

        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()

        # Final embedding layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        # GCN Layer 1
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        # GCN Layer 2
        x = self.conv2(x, edge_index)
        x = self.relu(x)

        # Global pooling
        x = torch.mean(x, dim=0)

        # Final embedding
        x = self.fc(x)

        return x


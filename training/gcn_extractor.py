
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GCNConv


class GCNExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=128):

        super().__init__(observation_space, features_dim)

        self.conv1 = GCNConv(2, 64)
        self.conv2 = GCNConv(64, 64)

        self.relu = nn.ReLU()

        self.fc = nn.Linear(64, features_dim)

  
    def forward(self, observations):

        batch_x = observations["x"]
        batch_edge_index = observations["edge_index"].long()

        embeddings = []

        batch_size = batch_x.shape[0]

        for i in range(batch_size):

            x = batch_x[i]
            edge_index = batch_edge_index[i]

            # GCN layer 1
            x = self.conv1(x, edge_index)
            x = self.relu(x)

            # GCN layer 2
            x = self.conv2(x, edge_index)
            x = self.relu(x)

            # Global pooling
            x = torch.mean(x, dim=0)

            # Final embedding
            x = self.fc(x)

            embeddings.append(x)

        embeddings = torch.stack(embeddings)

        return embeddings


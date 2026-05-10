
from env.grid_env import GridEnv
from training.gcn_policy import GCNFeatureExtractor

env = GridEnv()

graph = env.get_graph_data()

model = GCNFeatureExtractor()

embedding = model(graph)

print("Embedding shape:", embedding.shape)
print(embedding)


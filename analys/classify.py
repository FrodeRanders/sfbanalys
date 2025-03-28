import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
import pandas as pd

# === Load data extracted from Neo4j ===
# nodes.csv: id,text
# edges.csv: source,target,type

nodes_df = pd.read_csv('nodes.csv')
edges_df = pd.read_csv('edges.csv')

# === Build text embeddings ===
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(nodes_df['text'].tolist(), convert_to_tensor=True)

# === Build edge index (PyG expects [2, num_edges] tensor) ===
edge_index = torch.tensor([
    edges_df['source'].tolist(),
    edges_df['target'].tolist()
], dtype=torch.long)

# === Create the graph ===
data = Data(x=embeddings, edge_index=edge_index)

print(data)
# Data(x=[num_nodes, 768], edge_index=[2, num_edges])

label_map = {
    "beskrivning": 0,
    "definition": 1,
    "regel": 2,
    "other": 3  # catch-all
}

nodes_df["kategori"] = nodes_df["kategori"].fillna("").str.strip()
nodes_df["kategori"] = nodes_df["kategori"].replace("", "other")
nodes_df["label"] = nodes_df["kategori"].map(label_map)
data.y = torch.tensor(nodes_df["label"].tolist(), dtype=torch.long)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GNN(768, 256, 3)  # For 3-class classification

# Now, do the rest :)

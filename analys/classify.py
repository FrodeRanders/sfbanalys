import torch
from numpy.distutils.system_info import dfftw_info
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report


# === Load data extracted from Neo4j ===
# nodes.csv: id,text
# edges.csv: source,target,type

# DataFrame nodes and edges
df_nodes = pd.read_csv('nodes.csv')
df_edges = pd.read_csv('edges.csv')

assert df_edges["source"].isnull().sum() == 0
assert df_edges["target"].isnull().sum() == 0

# === Build text embeddings ===
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(df_nodes['text'].tolist(), convert_to_tensor=True)

# === Build edge index (PyG expects [2, num_edges] tensor) ===
edge_index = torch.tensor([
    df_edges["source"].astype(int).tolist(),
    df_edges["target"].astype(int).tolist()
], dtype=torch.long)

# === Create the graph ===
data = Data(x=embeddings, edge_index=edge_index)

print(data)
# Data(x=[num_nodes, 768], edge_index=[2, num_edges])
# i.e. Data(x=[2959, 768], edge_index=[2, 69348])

label_map = {
    "beskrivning": 0,
    "definition": 1,
    "regel": 2,
    "annat": -1  # catch-all
}

df_nodes["kategori"] = df_nodes["kategori"].fillna("").str.strip()
df_nodes["kategori"] = df_nodes["kategori"].replace("", "annat")
df_nodes["label"] = df_nodes["kategori"].map(label_map)
data.y = torch.tensor(df_nodes["label"].tolist(), dtype=torch.long)

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

# Create a Train/Validation Mask
# Only use nodes with known labels (i.e. label != -1)
labeled_nodes = (data.y != -1).nonzero(as_tuple=True)[0]

train_idx, val_idx = train_test_split(
    labeled_nodes.tolist(), test_size=0.2, random_state=42
)

train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[train_idx] = True

val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask[val_idx] = True

# Define Loss, Optimizer, Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
    return acc

for epoch in range(1, 201):
    loss = train()
    train_acc = evaluate(train_mask)
    val_acc = evaluate(val_mask)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# Print confusion matrix
print("-"*80)
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out[val_mask].argmax(dim=1).cpu()
    true = data.y[val_mask].cpu()

print(confusion_matrix(true, pred))
print(classification_report(true, pred, target_names=["beskrivning", "definition", "regel"]))


# Predict on all nodes, tagging the entire graph
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    predicted_labels = logits.argmax(dim=1).cpu().numpy()
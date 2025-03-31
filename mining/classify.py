import json
import numpy as np
from embed import embed
from cluster import cluster, plot_clusters

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

file_path = "../sfbreader/data/SFB-flat.json"

def main():
    texts = []
    embeddings = []

    # Read the JSON data and create embedding
    print("Skapa inbäddning...")
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        texts, embeddings = embed(data)

    print("Klustra...")
    np_embeddings = np.vstack(embeddings)  # shape: (n_samples, 384)

    labels, cluster_ids, topic_dict, representatives = cluster(texts, np_embeddings)

    plot_clusters(labels, cluster_ids, topic_dict, representatives, data, texts, np_embeddings)

if __name__ == "__main__":
    main()

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import umap.umap_ as umap
import hdbscan
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

file_path = "../sfbreader/data/SFB-flat.json"
model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")


def plot_clusters(embeddings_2d, labels, title="UMAP + HDBSCAN Clusters", filename=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import seaborn as sns

    unique_labels = np.unique(labels)
    palette = sns.color_palette("hls", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

    colors = [color_map[label] if label != -1 else (0.6, 0.6, 0.6) for label in labels]

    plt.figure(figsize=(10, 8))
    plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=colors,
        s=10,
        alpha=0.7
    )
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Lagrat till: {filename}")

    plt.show()

def main():
    texts = []
    embeddings = []

    # Read the JSON data from the local file
    print("Ladda data och skapa inbäddning...")
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

        #     "lag": "Socialförsäkringsbalk (2010:110)",
        #     "avdelning": "B FAMILJEFÖRMÅNER",
        #     "underavdelning": "II Graviditetspenning och föräldrapenningsförmåner",
        #     "kapitel": "12",
        #     "kapitel_namn": "Föräldrapenning",
        #     "paragraf_rubrik": "Rätten till föräldrapenning",
        #     "paragraf_underrubrik": "Allmänna bestämmelser",
        #     "paragraf": "3",
        #     "referens": "Lag (2013:999)",
        #     "stycke": 1,
        #     "text": "För rätt till föräldrapenning..."

        for record in data:
            #
            #input_text = f"[AVDELNING {record['avdelning']}]"
            #input_text = f"{input_text} [UNDERAVDELNING {record['underavdelning']}]"

            #kapitel_namn = record["kapitel_namn"]
            #if kapitel_namn:
            #    input_text = f"{input_text} [KAPITEL {kapitel_namn}]"

            input_text = ""
            paragraf_rubrik = record.get("paragraf_rubrik", None)
            if paragraf_rubrik:
                input_text = f"{input_text} [RUBRIK {paragraf_rubrik}]"

            paragraf_underrubrik = record.get("paragraf_underrubrik", None)
            if paragraf_underrubrik:
                input_text = f"{input_text} [UNDERRUBRIK {paragraf_underrubrik}]"

            input_text = f"{input_text} {record['text']}",
            embedding = model.encode(input_text, convert_to_numpy=True)

            #
            texts.append(input_text)
            embeddings.append(embedding[0])  # grab the vector, drop the extra dimension


    # Fit K-Means
    print("Klustring...")
    np_embeddings = np.array(embeddings)

    umap_model = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine')
    umap_embeddings = umap_model.fit_transform(np_embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
    labels = clusterer.fit_predict(umap_embeddings)

    assert len(labels) == len(embeddings) == len(texts)

    # For each cluster (excluding noise), pick a representative text
    cluster_ids = np.unique(labels)
    cluster_ids = [cid for cid in cluster_ids if cid != -1]  # exclude noise

    representatives = []

    for cluster_id in cluster_ids:
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            representatives.append(None)
            continue

        cluster_emb = np_embeddings[cluster_indices]
        centroid_vec = np.mean(cluster_emb, axis=0)  # shape: (embedding_dim,)
        centroid_vec_2d = centroid_vec.reshape(1, -1)

        similarities = cosine_similarity(centroid_vec_2d, cluster_emb)[0]
        best_local_idx = np.argmax(similarities)
        global_text_index = cluster_indices[best_local_idx]

        rep_text = texts[global_text_index]
        representatives.append(rep_text)

    topic_dict = {}
    for idx, label_id in enumerate(labels):
        if label_id not in topic_dict:
            topic_dict[label_id] = []
        topic_dict[label_id].append(texts[idx])

    # `topic_dict` is a dict with keys = cluster_id, values = list of texts
    # `representatives` is a list where representatives[c] is the text for cluster c
    n_noise = np.sum(labels == -1)
    print(f"Antal uteliggare (brus): {n_noise}")

    for idx, cluster_id in enumerate(cluster_ids):
        texter = topic_dict[cluster_id]
        print(f"Kluster: {cluster_id}\n")
        rep_text = representatives[idx]
        if rep_text is not None:
            print(f" * Kluster-center: {rep_text}\n")
        for text in texter:
            print(f" - {text}")
        print("\n")

    # Generate 2D UMAP embeddings
    umap_model_2d = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine')
    umap_embeddings_2d = umap_model_2d.fit_transform(np_embeddings)

    # Save plot to file (PNG, PDF, SVG, etc.)
    plot_clusters(umap_embeddings_2d, labels, filename="kluster.png")


if __name__ == "__main__":
    main()

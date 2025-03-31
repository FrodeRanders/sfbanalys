import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import umap.umap_ as umap
import hdbscan
from plot import interactive_plot_clusters
from explain import generate_labels_for_clusters


def cluster(texts, np_embeddings):
    umap_model = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine')
    umap_embeddings = umap_model.fit_transform(np_embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
    labels = clusterer.fit_predict(umap_embeddings)

    assert len(labels) == len(np_embeddings) == len(texts)

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

    return labels, cluster_ids, topic_dict, representatives

def plot_clusters(labels, cluster_ids, topic_dict, representatives, data, texts, np_embeddings):

    print("-"*80)
    cluster_labels = generate_labels_for_clusters(topic_dict)

    for cluster_id, texter in topic_dict.items():
        print(f"\nKluster {cluster_id} – {cluster_labels[cluster_id]}\n")
        for text in texter[:5]:
            print(f" - {text}")

    print("-"*80)
    for idx, cluster_id in enumerate(cluster_ids):
        texter = topic_dict[cluster_id]
        print(f"Kluster: {cluster_id}\n")
        rep_text = representatives[idx]
        if rep_text is not None:
            print(f"o> Kluster-center: {rep_text}\n")
        for text in texter:
            print(f"*> {text}")
        print("\n")

    # Generate 2D UMAP embeddings
    umap_model_2d = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine')
    umap_embeddings_2d = umap_model_2d.fit_transform(np_embeddings)

    # Plot and save to file
    #plot_clusters(umap_embeddings_2d, labels, filename="kluster.png")

    extra_info = {
        "kapitel": [r["kapitel"] for r in data],
        "paragraf": [r["paragraf"] for r in data],
        "stycke": [r["stycke"] for r in data]
        #"rubrik": [r.get("paragraf_rubrik", "") for r in data]
    }

    interactive_plot_clusters(umap_embeddings_2d, labels, texts, extra_info, "kluster.html")

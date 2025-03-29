import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

file_path = "../sfbreader/data/SFB-flat.json"
model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")

# Prova med 20 kluster
K = 20


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

    kmeans = KMeans(n_clusters=K, n_init="auto")
    labels = kmeans.fit_predict(np_embeddings)
    assert len(labels) == len(embeddings) == len(texts)

    # For each labeled cluster, pick a representative text
    representatives = []
    for cluster_id in range(K):
        # 'cluster_indices' is an array of integers, which are indices of the
        # texts in this particular cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            # No texts in this cluster (edge case if KMeans had an empty cluster)
            representatives.append(None)
            continue

        # Grab centroid vector
        centroid_vec = kmeans.cluster_centers_[cluster_id]  # shape (embedding_dim,)

        # Only compare centroid with embeddings of current cluster
        cluster_emb = np_embeddings[cluster_indices]  # shape: (n_cluster_docs, 384)
        centroid_vec_2d = centroid_vec.reshape(1, -1)

        # Compute similarities with *just* the cluster embeddings
        similarities = cosine_similarity(centroid_vec_2d, cluster_emb)[0]

        # Now this index is local to cluster_emb
        best_local_idx = np.argmax(similarities)

        # Get the global index from cluster_indices
        global_text_index = cluster_indices[best_local_idx]

        # E.g., get the corresponding text
        rep_text = texts[global_text_index]
        representatives.append(rep_text)

    topic_dict = {}
    for idx, label_id in enumerate(labels):
        if label_id not in topic_dict:
            topic_dict[label_id] = []
        topic_dict[label_id].append(texts[idx])

    # `topic_dict` is a dict with keys = cluster_id, values = list of texts
    # `representatives` is a list where representatives[c] is the text for cluster c
    for cluster_id, texter in topic_dict.items():
        print(f"Kluster: {cluster_id}\n")
        rep_text = representatives[cluster_id]
        if rep_text is not None:
            print(f" * Kluster-center: {rep_text}\n")

        for text in texter:
            print(f" - {text}")

        print("\n")

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

def truncate_lines(text: str, max_len: int = 40, max_lines: int = None) -> str:
    lines = text.split("\n")
    if max_lines:
        lines = lines[:max_lines]
    truncated = [line[:max_len] + ("…" if len(line) > max_len else "") for line in lines]
    return "<br>".join(truncated)

def interactive_plot_clusters(embeddings_2d, labels, texts, extra_info=None, filename=None):
    texts_html = [truncate_lines(t, max_len=80, max_lines=10) for t in texts]

    # Prepare DataFrame
    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "cluster": labels,
        "text": texts_html
    })

    # Add extra columns if provided
    if extra_info:
        for key, values in extra_info.items():
            df[key] = values

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_data=["cluster", "text"] + (list(extra_info.keys()) if extra_info else []),
        title="Interactive UMAP + HDBSCAN Clusters",
        color_continuous_scale="Viridis"
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7))

    if filename:
        fig.write_html(filename)
        print(f"Lagrat till: {filename}")
    else:
        fig.show()

def plot_clusters(embeddings_2d, labels, title="UMAP + HDBSCAN Clusters", filename=None):
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
    else:
        plt.show()


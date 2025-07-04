import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

def center_clusters(pos, clusters, weight=0.3):
    """
    Move clusters closer to the center of the layout.
    Args:
        pos: dict from spring_layout (node_id -> (x, y))
        clusters: list of sets of node_ids
        weight: float [0, 1] – how strongly to pull each cluster toward center
    """
    center = np.mean(list(pos.values()), axis=0)

    for cluster in clusters:
        members = list(cluster)
        cluster_center = np.mean([pos[n] for n in members], axis=0)
        offset = center - cluster_center
        for n in members:
            pos[n] = pos[n] + weight * offset
    return pos

def visualize_clusters(clusters, all_docs, max_docs_per_cluster=50, seed=42):
    """
    Visualizes document clusters using NetworkX.
    Each cluster is shown as a separate color.
    """
    random.seed(seed)
    G = nx.Graph()

    cluster_colors = {}
    color_palette = plt.cm.get_cmap('tab10', len(clusters))

    for idx, cluster in enumerate(clusters):
        # Limit cluster size for clarity
        members = list(cluster)
        if len(members) > max_docs_per_cluster:
            members = random.sample(members, max_docs_per_cluster)

        # Add edges between all pairs (clique)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                G.add_edge(members[i], members[j])

        for node in members:
            cluster_colors[node] = color_palette(idx)

    # Draw with spring layout
    pos = nx.spring_layout(G, seed=seed, k=0.6, iterations=100)
    pos = center_clusters(pos, clusters, weight=0.6)
    node_colors = [cluster_colors[n] for n in G.nodes]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=120, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.title(f"LSH-Based Clustering for deduplicating Documents: {len(clusters)} unique clusters identified", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
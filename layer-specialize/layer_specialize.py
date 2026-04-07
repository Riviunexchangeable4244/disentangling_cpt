#!/usr/bin/env python3
"""
Select interface and semantic hub layers using K-means clustering.
Clusters layers by accuracy into k groups (default k=2).
Lower accuracy cluster = interface layers, higher accuracy = semantic layers.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from sklearn.cluster import KMeans


def select_layers_by_kmeans(json_path: Path) -> Dict[str, List[int]]:
    """
    Cluster layers into k groups based on accuracy.
    
    Args:
        json_path: Path to JSON file with layer_accuracy field
        k: Number of clusters (default: 2)
    
    Returns:
        Dictionary with interface_layers and semantic_layers
    """
    data = json.loads(json_path.read_text())
    
    # Extract accuracies and indices, skipping layer 0
    layer_acc = np.array([acc for idx, acc in data["layer_accuracy"] if idx > 0]).reshape(-1, 1)
    layer_idxs = [idx - 1 for idx, acc in data["layer_accuracy"] if idx > 0]

    # Cluster layers
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(layer_acc)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_.flatten()

    # Lower accuracy = interface, higher accuracy = semantic
    low_cluster = np.argmin(cluster_centers)
    high_cluster = np.argmax(cluster_centers)

    interface_layers = [layer_idxs[i] for i, lbl in enumerate(labels) if lbl == low_cluster]
    semantic_layers = [layer_idxs[i] for i, lbl in enumerate(labels) if lbl == high_cluster]

    return {
        "interface_layers": interface_layers,
        "semantic_layers": semantic_layers,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cluster model layers into interface and semantic groups using K-means"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSON file with layer_accuracy data"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output JSON file (default: stdout)"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")
    
    layers_dict = select_layers_by_kmeans(args.input)
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(layers_dict, f, indent=4)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(layers_dict, indent=4))


if __name__ == "__main__":
    main()
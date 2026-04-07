# Layer Selection via K-Means Clustering

Automatically identify interface and semantic hub layers in multilingual models using unsupervised clustering on sentence retrieval accuracy scores.

## Overview

This tool categorizes model layers into two groups based on their cross-lingual alignment performance:

- **Interface Layers**: Low retrieval accuracy - handle language-specific features (syntax, morphology, script)
- **Semantic Hub Layers**: High retrieval accuracy - contain language-agnostic semantic representations

The categorization uses K-means clustering on per-layer sentence retrieval scores, providing an automated way to identify which layers specialize in different aspects of language processing.

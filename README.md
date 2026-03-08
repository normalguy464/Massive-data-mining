# Massive-data-mining

# 🌐 Massive Data Mining: Reddit Community Structure & User Migration

## 📖 About The Project
This repository contains the source code for an end-to-end Big Data pipeline and network analysis project focusing on the massive **Reddit Pushshift Dataset**. 

The project is divided into two main phases:
1. **Data Engineering (ETL):** Extracting, cleaning, and transforming terabytes of raw `.zst` compressed JSON dumps from torrents into highly optimized, columnar **Parquet** files hosted on Hugging Face.
2. **Data Mining & Network Analysis:** Applying graph-theory and data mining techniques to the cleaned dataset to analyze user behavior, community clustering, and migration patterns across the Reddit ecosystem.

## 📚 Theoretical Foundation & Objectives
The analytical phase of this project is heavily inspired by and builds upon the following academic research:

1. **Exploring Reddit Community Structure: Bridges, Gateways and Highways** *(Electronics, 2024)*
   - **Objective:** We aim to construct user-subreddit bipartite graphs to analyze the macro-structure of Reddit. By evaluating node centrality and edge weights, we will identify which subreddits act as "bridges" (connecting isolated communities) or "gateways" (onboarding users to broader topics).
2. **Multi-Scale User Migration on Reddit** *(ICWSM, 2021)*
   - **Objective:** We will track overlapping user bases to understand how users migrate between communities. A key focus will be analyzing macro and micro-level migration patterns, particularly observing how users disperse after platform moderation interventions (e.g., when a subreddit is banned).

## 🗂️ Repository Structure (Current Phase)
Currently, the repository contains the **Data Engineering** scripts used to build the foundational dataset.

* `Data processing/`
  * 📄 `torrent_download.py`: The core ETL engine. It asynchronously streams downloaded `.zst` files from qBittorrent, decompresses them, parses JSON lines using `orjson`, normalizes data types, handles missing values, and securely batches uploads Parquet files to the Hugging Face Hub.
  * 📄 `filtering_files_pushshift.py`: A smart filtering script. It applies Regex and CamelCase logic to identify and remove highly sensitive/NSFW communities based on a keyword list, ensuring the dataset's safety.
  * 📄 `delete_file_repo.py`: The repository janitor. It synchronizes local trash logs (empty/invalid files) with the remote Hugging Face repo and cleans up corrupted Parquet files (e.g., missing magic bytes) caused by network interruptions.

## 🚀 Roadmap (Next Steps)
Our team is moving towards the Network Analysis phase. Upcoming scripts will include:
- [ ] **Graph Construction:** Building nodes (Users, Subreddits) and edges (Interactions, Co-authorship) using `NetworkX` or `PySpark`.
- [ ] **Community Detection:** Applying clustering algorithms (e.g., Louvain, Leiden) to group similar subreddits.
- [ ] **Centrality Analysis:** Calculating Betweenness and Eigenvector centrality to find "Bridge" communities.
- [ ] **Migration Tracking:** Time-series analysis of user activity before and after specific subreddit bans.

## 🛠️ Prerequisites & Setup
To run the data processing scripts, you need:
- Python 3.8+
- `qbittorrent-api` (for torrent client integration)
- `huggingface_hub` & `datasets` (for cloud storage)
- `pyarrow`, `pandas`, `zstandard`, `orjson` (for high-speed data manipulation)

Create a `.env` file in the root directory with your Hugging Face token:
```env
HF_TOKEN=your_hf_access_token_here

# Massive-data-mining

# 🌐 Massive Data Mining: Multi-Scale User Migration on Reddit

## 📖 About The Project
This repository contains the source code for an end-to-end Big Data pipeline and data mining project focusing on the massive **Reddit Pushshift Dataset**. 

The project is divided into two main phases:
1. **Data Engineering (ETL):** Extracting, cleaning, and transforming terabytes of raw `.zst` compressed JSON dumps from torrents into highly optimized, columnar **Parquet** files hosted on Hugging Face.
2. **Data Mining & Migration Analysis:** Tracking and analyzing the dynamics of user migration across the Reddit ecosystem, particularly in response to platform interventions (e.g., subreddit bans or quarantines).

When a subreddit is banned or shut down, its user base does not simply disappear; they migrate to other active communities. This project aims to map these "digital migrations" across two specific scales:
* **Macro-Scale (Community Level):** By building transition matrices and calculating user overlap over time-windows, we identify the destination subreddits that act as "refuges" for displaced users. We analyze how the structural overlap between communities shifts before and after an intervention event.
* **Micro-Scale (Individual Level):** We track the activity levels (e.g., posting frequency) of the affected users to determine behavioral changes. Does a community ban suppress a user's engagement, or does it radicalize and increase their activity in a new environment?

## 🗂️ Repository Structure (Current Phase)
Currently, the repository contains the **Data Engineering** scripts used to build the foundational dataset required for this heavy temporal analysis.

* `Data processing/`
  * 📄 `torrent_download.py`: The core ETL engine. It asynchronously streams downloaded `.zst` files from qBittorrent, decompresses them, parses JSON lines using `orjson`, normalizes data types, handles missing values, and securely batches uploads Parquet files to the Hugging Face Hub.
  * 📄 `filtering_files_pushshift.py`: A smart filtering script. It applies Regex and CamelCase logic to identify and remove highly sensitive/NSFW communities based on a keyword list, ensuring the dataset's safety.
  * 📄 `delete_file_repo.py`: The repository janitor. It synchronizes local trash logs (empty/invalid files) with the remote Hugging Face repo and cleans up corrupted Parquet files (e.g., missing magic bytes) caused by network interruptions.

## 🚀 Roadmap (Next Steps)
With the foundational dataset cleaned and optimized, our team is moving towards the **Data Mining** phase. Upcoming scripts will include:
- [ ] **Temporal Data Windowing:** Splitting the dataset into pre-intervention and post-intervention timeframes based on the `created_utc` timestamp.
- [ ] **Macro Migration Mapping:** Calculating Jaccard similarity and user overlap metrics to trace the flow of users between subreddits.
- [ ] **Micro Activity Tracking:** Aggregating individual user engagement metrics to analyze behavioral shifts.
- [ ] **Visualization:** Plotting migration flows (e.g., Sankey diagrams) and activity distributions over time.

## 🛠️ Prerequisites & Setup
To run the data processing scripts, you need:
- Python 3.8+
- `qbittorrent-api` (for torrent client integration)
- `huggingface_hub` & `datasets` (for cloud storage)
- `pyarrow`, `pandas`, `zstandard`, `orjson` (for high-speed data manipulation)

- hihi

Create a `.env` file in the root directory with your Hugging Face token:
```env
HF_TOKEN=your_hf_access_token_here

hehehe

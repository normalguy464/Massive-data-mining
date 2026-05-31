# Massive Data Mining Reddit Semantic Graph

Phân tích hệ sinh thái subreddit từ Reddit Pushshift Dataset bằng xử lý dữ liệu lớn, embedding văn bản, graph mining, community detection và trực quan hóa.

---

## Mục Lục

1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Câu Hỏi Phân Tích](#2-câu-hỏi-phân-tích)
3. [Luồng Xử Lý Dữ Liệu](#3-luồng-xử-lý-dữ-liệu)
4. [Bảng Số Liệu Tổng Hợp](#4-bảng-số-liệu-tổng-hợp)
5. [Kết Quả Graph và Community](#5-kết-quả-graph-và-community)
6. [Bridge, Gateway, Highway và PageRank Nibble](#6-bridge-gateway-highway-và-pagerank-nibble)
7. [Cấu Trúc Workspace](#7-cấu-trúc-workspace)
8. [Cách Chạy Lại Các Thành Phần Chính](#8-cách-chạy-lại-các-thành-phần-chính)
9. [Cấu Hình Môi Trường](#9-cấu-hình-môi-trường)
10. [Lưu Ý Khi Phát Triển Tiếp](#10-lưu-ý-khi-phát-triển-tiếp)

---

## 1. Tổng Quan Dự Án

Dự án xây dựng hai dạng graph từ dữ liệu Reddit:

- **Graph tương đồng nội dung** — Node là subreddit, cạnh biểu diễn cosine similarity giữa các vector embedding tiêu đề bài viết.
- **Graph crosspost** — Cạnh biểu diễn số lần bài viết từ một subreddit được crosspost sang subreddit khác.

Từ hai graph này, dự án phát hiện community, đặt tên bằng LLM, phân tích bridge/gateway/highway và chạy PageRank Nibble để tìm cụm cục bộ.

---

## 2. Câu Hỏi Phân Tích

- Subreddit nào có nội dung giống nhau nhất theo embedding tiêu đề?
- Hệ sinh thái subreddit tách thành những community chủ đề nào?
- Community nào lớn, nhỏ, trung tâm hoặc ngoại vi?
- Subreddit nào đóng vai trò **bridge** (cầu nối), **gateway** (cửa ngõ) hoặc **highway** (tuyến lặp) giữa các community?
- PageRank Nibble tìm được cụm cục bộ nào quanh một seed — kích thước và conductance ra sao?

---

## 3. Luồng Xử Lý Dữ Liệu

### Giai Đoạn 1 — Thu Thập và Làm Sạch Dữ Liệu
Dữ liệu Pushshift `.zst` được tải, giải nén, parse JSON line, lọc không hợp lệ và ghi thành Parquet. File trùng, nhạy cảm và hỏng được loại bỏ trước khi upload Hugging Face.

### Giai Đoạn 2 — Chọn Dữ Liệu Phân Tích
Submission được lọc theo subreddit, score, số comment, upvote_ratio và độ dài title. Mỗi subreddit giữ tối đa 1.000 bài viết tốt nhất để giảm nhiễu và chi phí embedding.

### Giai Đoạn 3 — Tạo Embedding và Tính Similarity
Title được chuẩn hóa và đưa vào DistilBERT (mean pooling). Cosine similarity giữa các vector subreddit tạo ra `result/subreddit_similarity_results.csv`. Khi cần mở rộng quy mô, notebook LSH dùng `BucketedRandomProjectionLSH` trong PySpark.

### Giai Đoạn 4 — Xây Graph và Phát Hiện Community
Louvain là thuật toán phát hiện community chính; Girvan-Newman dùng để đối chiếu. Kết quả ghi vào `result/community_result.csv`.

### Giai Đoạn 5 — Phân Tích Vai Trò Graph
Tính bridge score, gateway score, highway extraction và PageRank Nibble (đánh giá bằng conductance).

### Giai Đoạn 6 — Trực Quan Hóa và Demo
Graph HTML bằng PyVis, ảnh thống kê bằng matplotlib, dashboard web bằng React/Vite.

---

## 4. Bảng Số Liệu Tổng Hợp

### Artefact Dữ Liệu Chính

| Artefact | Quy Mô | Ý Nghĩa |
|---|---|---|
| `result/subreddit_similarity_results.csv` | 3.225.710 dòng | Cặp subreddit và Similarity_Score |
| `graph-weight.csv` | 1.793.660 dòng | Cạnh crosspost có trọng số |
| `result/community_result.csv` | 3.256 dòng | Mapping subreddit → community_id, community_size |
| `result/cluster_names.csv` | 91 community | Tên community đặt bằng LLM |
| `result/bridge_result.csv` | 864 dòng | Subreddit có vai trò bridge |
| `result/gateway_result.csv` | 3.169 dòng | Gateway score của subreddit trong community |
| `result/highway_result.csv` | 80 dòng | Highway xếp hạng theo tần suất |
| `result/pagerank_nibble_result.csv` | 180 dòng | Node trong cụm PageRank Nibble theo seed |
| `result/pagerank_nibble_summary.csv` | 12 seed | Conductance, cut, volume và PPR mass |
| `result/subreddit_content_map.csv` | 28 dòng | Nội dung subreddit crawl hoặc suy luận từ cluster |

### Phân Phối Similarity_Score

| Chỉ Số | Giá Trị |
|---|---|
| Số cặp | 3.225.710 |
| Min | 0.3000 |
| Mean | 0.4566 |
| Median | 0.4305 |
| P75 | 0.5254 |
| P90 | 0.6276 |
| P95 | 0.6929 |
| P97 | 0.7361 |
| P99 | 0.8199 |
| Max | 0.9989 |

Score tập trung ở vùng 0.3–0.5. Ngưỡng P97 (~0.74) được dùng làm cạnh mạnh trong nhiều bước graph mining. Vùng trên 0.9 thường là subreddit cùng chủ đề hoặc biến thể tên.

### Top Cặp Similarity Cao Nhất

| # | Subreddit A | Subreddit B | Score |
|---|---|---|---|
| 1 | 196 | 19684 | 0.9989 |
| 2 | AliexpressCouponFind | AliexpressCoupons2021 | 0.9986 |
| 3 | AxieInfinityScholar | AxieScholarshipsPH | 0.9971 |
| 4 | BSCMoonShots | AllCryptoBets | 0.9967 |
| 5 | 19684 | 197 | 0.9966 |
| 6 | CoinSales | Coins4Sale | 0.9965 |
| 7 | CouponCodeExplore | CouponCodeNews | 0.9961 |
| 8 | AbsoluteWeapons | AbsoluteWeaponz | 0.9961 |

Các cặp điểm cao thường có tên gần giống hoặc cùng miền nội dung hẹp — cho thấy embedding nắm được cả quan hệ ngữ nghĩa lẫn quan hệ đặt tên.

### Tóm Tắt Graph Crosspost

| Chỉ Số | Giá Trị |
|---|---|
| Số cạnh | 1.793.660 |
| Số subreddit | 28.252 |
| Tổng trọng số | 10.150.187 |
| Weight trung bình | 5.66 |

### Top Cạnh Crosspost Theo Weight

| # | Source | Target | Weight |
|---|---|---|---|
| 1 | hackernews | patient_hackernews | 54.135 |
| 2 | france | discussion_patiente | 27.956 |
| 3 | mexico_politics | Mexico_News | 26.511 |
| 4 | coins | MetalsOnReddit | 21.348 |
| 5 | Silverbugs | MetalsOnReddit | 18.482 |
| 6 | boardgames | PersonalizedGameRecs | 18.416 |
| 7 | mexico | mejico | 17.308 |
| 8 | Gold | MetalsOnReddit | 16.594 |

Weight cao phản ánh quan hệ mirror, repost hoặc community vận hành song song — khác với graph similarity vốn dựa trên nội dung.

---

## 5. Kết Quả Graph và Community

### Tóm Tắt Community Detection

| Chỉ Số | Giá Trị |
|---|---|
| Số subreddit đã gán | 3.256 |
| Số community | 91 |
| Community lớn nhất | 605 subreddit |
| Community nhỏ nhất | 2 subreddit |
| Kích thước trung bình | 35.78 |
| Median kích thước | 3 |

Median = 3 trong khi max = 605 cho thấy phân phối lệch mạnh: vài cụm chủ đề rất rộng và nhiều cụm nhỏ chuyên biệt, phù hợp cấu trúc thực tế của Reddit.

### Phân Bố Kích Thước Community

| Size | Số Community |
|---|---|
| 2 | 41 |
| 3 | 16 |
| 4 | 6 |
| 5 | 3 |
| 6 | 4 |
| 7 | 2 |
| 13 | 1 |
| 14 | 1 |
| 21 | 1 |
| 25 | 1 |

### Top 12 Community Lớn Nhất

| # | Community ID | Số Subreddit | Tên |
|---|---|---|---|
| 1 | 0 | 605 | Anime & Pop Culture |
| 2 | 29 | 373 | Game Chiến Tranh & Anime |
| 3 | 9 | 331 | Hookup & Attraction |
| 4 | 50 | 280 | Ô Tô và Công Nghệ |
| 5 | 24 | 228 | Học Tập & Địa Lý |
| 6 | 46 | 195 | Mental Health & Support |
| 7 | 75 | 190 | Thú Cưng & Trò Chơi |
| 8 | 21 | 164 | Politics & Conspiracy |
| 9 | 7 | 163 | Nostalgia & Culture |
| 10 | 3 | 106 | Crypto & Finance |
| 11 | 10 | 97 | Âm Nhạc Đa Thể Loại |
| 12 | 26 | 79 | Cannabis & Fitness |

---

## 6. Bridge, Gateway, Highway và PageRank Nibble

### Tóm Tắt Các Lớp Phân Tích Vai Trò

| Phân Tích | Quy Mô | Ý Nghĩa |
|---|---|---|
| Bridge | 864 dòng | Node kết nối giữa các community |
| Gateway | 3.169 dòng | Node là cửa ngõ trong community |
| Highway | 80 dòng | Subpath lặp lại nhiều trên shortest paths |
| PageRank Nibble | 12 seed | Cụm cục bộ đánh giá bằng conductance |

### Top Bridge Theo Bridge_Score

| # | Subreddit | Source → Target | Score |
|---|---|---|---|
| 1 | ConservativeKiwi | 48 → 0 | 0.0674 |
| 2 | 2islamist4you | 33 → 0 | 0.0577 |
| 3 | Chriswatts | 51 → 17 | 0.0438 |
| 4 | Aliexpress | 20 → 0 | 0.0335 |
| 5 | BehindTheClosetDoor | 20 → 0 | 0.0334 |
| 6 | CatholicMemes | 15 → 0 | 0.0298 |
| 7 | CNNmemes | 42 → 0 | 0.0197 |
| 8 | AdviceAtheists | 15 → 0 | 0.0165 |

### Top Gateway Theo Điểm Chuẩn Hóa

| # | Subreddit | Community | Normalized | Raw Score |
|---|---|---|---|---|
| 1 | AnimeFunny | 0 | 1.0000 | 0.001106 |
| 2 | CoinWithUs | 3 | 1.0000 | 0.000125 |
| 3 | ARK_pc | 49 | 1.0000 | 0.0000477 |
| 4 | AnimeMeme | 6 | 1.0000 | 0.001263 |
| 5 | AI_Girl | 7 | 1.0000 | 0.000682 |
| 6 | CostaRicaTravel | 8 | 1.0000 | 0.000187 |
| 7 | CampLejeuneGW | 9 | 1.0000 | 0.000461 |
| 8 | CartoonGangsters | 10 | 1.0000 | 0.000906 |

> Điểm chuẩn hóa theo từng community, nên nhiều community có thể đạt 1.0. Khi so sánh liên community, nên xem thêm `gateway_score` gốc và `community_size`.

### Top Highway Theo Tần Suất

| # | Length | Nodes | Occurrences |
|---|---|---|---|
| 1 | 2 | ConservativeKiwi → CapeIndependence | 22 |
| 1 | 3 | BlackHair → Barber → BeardAdvice | 17 |
| 1 | 4 | AimeLeonDore → Athleta_gap → BlackestFridayDeals → BestThings | 14 |
| 1 | 5 | AdoptMeTrading → AdoptMeTradingRoblox → BloxFruitsTrades → BalisongSale → AVexchange | 9 |
| 2 | 2 | 2islamist4you → Afghan | 21 |
| 2 | 3 | Athleta_gap → BlackestFridayDeals → BestThings | 16 |
| 2 | 4 | AmazonWFShoppers → Aliexpress → Buyee → AmazonSeller | 12 |
| 2 | 5 | ATBandATGcommunity → ButchSelfies → BlackHair → Barber → BeardAdvice | 9 |

### PageRank Nibble — Conductance Thấp Nhất

| # | Seed | Cluster Size | Conductance | PPR Mass | Top Subreddits |
|---|---|---|---|---|---|
| 1 | CapeIndependence | 3 | 0.1798 | 0.7772 | CapeIndependence, AfricaVoice, Africa |
| 2 | CostaRicaTravel | 8 | 0.2939 | 0.7884 | CostaRicaTravel, CaminoDeSantiago, Aruba |
| 3 | BlackHair | 6 | 0.3573 | 0.5803 | BlackHair, Barber, 360Waves |
| 4 | ConservativeKiwi | 14 | 0.3575 | 0.5792 | ConservativeKiwi, CapeIndependence, AfricaVoice |
| 5 | Chriswatts | 8 | 0.3961 | 0.5339 | Chriswatts, Columbine, AshaDegree |
| 6 | 2islamist4you | 25 | 0.4493 | 0.5243 | 2islamist4you, Afghan, AfghanCivilwar |
| 7 | Aliexpress | 12 | 0.4923 | 0.6760 | Aliexpress, Buyee, Bricklink |
| 8 | CoinWithUs | 80 | 0.5992 | 0.5099 | CoinWithUs, CoinstarFinds, Bovada |

Conductance càng thấp thì cụm càng tách biệt. Seed có cluster size lớn thường có conductance cao hơn do mở rộng ra nhiều vùng liên kết.

---

## 7. Cấu Trúc Workspace

### Thư Mục Chính

| Thư Mục | Vai Trò |
|---|---|
| `Data Processing` | ETL Pushshift, lọc file, upload/kiểm tra Hugging Face |
| `processing-local` | Pipeline local: lọc bài, chuẩn hóa title, embedding, LSH |
| `Build_Graph_base_distributed` | Notebook phân tán trên Azure/Databricks/Colab |
| `community_detection` | Louvain, Girvan-Newman, bridge, gateway, highway |
| `result` | CSV, JSON và ảnh kết quả |
| `Visualize` | HTML graph và ảnh trực quan hóa |
| `web-demo` | Dashboard React/Vite trình bày graph và thống kê |

### File Python Quan Trọng

| File | Vai Trò |
|---|---|
| `chat_with_llm_api.py` | Helper gọi LLM API và trích nội dung trả về |
| `name_clusters.py` | Đặt tên community bằng LLM |
| `plot_similarity_histogram.py` | Vẽ histogram Similarity_Score |
| `result/analyze_community_result.py` | Thống kê community và xuất ảnh tổng hợp |
| `result/pagerank_nibble.py` | Chạy Approximate PPR và sweep cut |
| `processing-local/scripts/craw_data.py` | Tải subset Parquet từ Hugging Face |
| `processing-local/scripts/crawl_subreddit_content.py` | Crawl mô tả subreddit cho web-demo |
| `web-demo/scripts/build-demo-data.mjs` | Build `demoData.json`, `communityIndex.json` và graph index |

### Notebook Chính

| Notebook | Vai Trò |
|---|---|
| `processing-local/notebook/filter_top_posts_per_subreddit.ipynb` | Lọc top bài theo chất lượng trong từng subreddit |
| `processing-local/notebook/prepare_title_embedding_input.ipynb` | Chuẩn hóa title và tạo input embedding |
| `processing-local/notebook/run_embed_in_kaggle.ipynb` | Chạy DistilBERT và tính cosine similarity |
| `processing-local/notebook/compute_lsh_similarity.ipynb` | Tính similarity xấp xỉ bằng PySpark LSH |
| `processing-local/notebook/community_detection.ipynb` | Chạy Louvain trên graph similarity |
| `community_detection/Girvan-Newman_community.ipynb` | Thử nghiệm Girvan-Newman và modularity |
| `community_detection/brigde_and_gateway.ipynb` | Tính bridge và gateway |
| `community_detection/highway_extraction.ipynb` | Trích xuất highway từ shortest paths |
| `Visualiza_graph.ipynb` | Tạo graph HTML và phân tích graph |
| `insight_visualize.ipynb` | Tạo biểu đồ insight và phân tích community |

---

## 8. Cách Chạy Lại Các Thành Phần Chính

| Tác Vụ | Lệnh |
|---|---|
| Cài phụ thuộc local | `pip install -r processing-local/requirements.txt` |
| Vẽ histogram similarity | `python plot_similarity_histogram.py result/subreddit_similarity_results.csv --min-score 0.3 --max-score 0.99 --output Visualize/similarity_histogram_0p3_0p99.png` |
| Tổng hợp community | `python result/analyze_community_result.py` |
| Đặt tên community | `python name_clusters.py --csv result/community_result.csv --out result/cluster_names.csv` |
| Chạy PageRank Nibble | `python result/pagerank_nibble.py --graph-csv result/subreddit_similarity_results.csv --community-csv result/community_result.csv --cluster-names-csv result/cluster_names.csv` |
| Crawl nội dung subreddit | `python processing-local/scripts/crawl_subreddit_content.py --top-similarity 16 --append` |
| Build data web-demo | `cd web-demo && npm install && npm run generate:data` |
| Chạy web-demo local | `cd web-demo && npm run dev -- --port 5173` |

---

## 9. Cấu Hình Môi Trường

| Biến | Dùng Cho |
|---|---|
| `HF_TOKEN` | Upload, list, xóa file trên Hugging Face |
| `LLM_API_URL` | Endpoint LLM cho `chat_with_llm_api.py` và `name_clusters.py` |
| `SHIELD_API_KEY` | API key khi gọi LLM |
| `LLM_MODEL` | Tên model mặc định |
| `REDDIT_CLIENT_ID` | OAuth Reddit khi crawl subreddit content |
| `REDDIT_CLIENT_SECRET` | OAuth Reddit khi crawl subreddit content |
| `AZURE_STORAGE_ACCOUNT_NAME` | Notebook Azure/Databricks |
| `AZURE_STORAGE_ACCOUNT_KEY` | Notebook Azure/Databricks |
| `AZURE_CONTAINER_NAME` | Container dữ liệu trên Azure |

> **Không commit credential thật vào repo.** Các notebook Azure/Databricks hiện có placeholder — cần thay bằng biến môi trường hoặc secret manager trước khi chạy lại.

---

## Kết Luận

Workspace bao gồm đầy đủ các lớp của một pipeline khai phá dữ liệu lớn: ETL Pushshift → lọc/chuẩn hóa → embedding → similarity → graph → community detection → đặt tên LLM → bridge/gateway/highway → PageRank Nibble → trực quan hóa HTML → dashboard React. Các bảng số liệu trong README ánh xạ trực tiếp đến các file kết quả hiện có trong workspace.

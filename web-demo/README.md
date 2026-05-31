# Reddit Semantic Graph Demo

React/Vite demo cho báo cáo khai phá dữ liệu lớn.

## Chạy local

```bash
npm install
npm run generate:data
npm run dev -- --port 5173
```

Web đọc dữ liệu từ `public/data/demoData.json`, được sinh từ các CSV kết quả trong repo:

- `result/community_result.csv`
- `result/cluster_names.csv`
- `result/bridge_result.csv`
- `result/gateway_result.csv`
- `result/highway_result.csv`
- `subreddit_similarity_results.csv`
- `result/subreddit_content_map.csv` (mapping `subreddit -> nội dung`, sinh bằng crawler bên dưới)

## Crawl nội dung subreddit

```bash
python ../processing-local/scripts/crawl_subreddit_content.py --top-similarity 16 --append
npm run generate:data
```

Nếu Reddit public endpoint trả `403`, cấu hình OAuth trước khi chạy lại:

```bash
set REDDIT_CLIENT_ID=...
set REDDIT_CLIENT_SECRET=...
python ../processing-local/scripts/crawl_subreddit_content.py --top-similarity 16 --append
```

Khi không crawl được từ Reddit, script tự fallback sang mô tả community để UI vẫn có nội dung hiển thị.

## Nội dung demo

Demo có 2 tab chính:

- `Graph`: gom toàn bộ graph hiện có và load trực tiếp các file HTML graph trong `public/graphs`.
- `Thống kê`: gom các bảng kết quả, metric, community, bridge, gateway, highway và thí nghiệm.

Danh sách graph nhẹ được đọc từ `public/graphs/index.json`. Các file HTML graph được copy từ root repo sang `public/graphs` khi chạy `npm run generate:data`.

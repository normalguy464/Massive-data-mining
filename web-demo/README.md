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

## Nội dung demo

Demo có 2 tab chính:

- `Graph`: gom toàn bộ graph hiện có, tải từng graph JSON theo lựa chọn và tự giới hạn cạnh khi graph lớn.
- `Thống kê`: gom các bảng kết quả, metric, community, bridge, gateway, highway và thí nghiệm.

Danh sách graph nhẹ được đọc từ `public/graphs/index.json`. Nội dung community/subreddit trong panel node được đọc từ `public/data/communityIndex.json`. Hai file này được sinh lại khi chạy `npm run generate:data`.

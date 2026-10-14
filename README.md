# Inventory Simulator MVP

MVP Streamlit dashboard theo `inventory_simulator_architecture_draft.md`. App dùng trực tiếp:

- `Data/df_detrend.parquet` (7.017.300 dòng, 5.650 M5 HOBBIES series)
- `Models/model_{store}_{dept}.pkl` (20 LightGBM Booster artifacts)

## Chạy

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

App được chia thành 7 workspace độc lập: Data Explorer, Forecast Explorer, Inventory Simulation, Strategy Comparison, Optimization, Strategy Statistics và Strategy Visualization. Mỗi tab có selector Store → Department → Item, tham số và nút Run riêng; không dùng một sidebar chung cho toàn bộ phân tích. Evaluation windows được cố định theo yêu cầu: optimization `d=1830..1913`, final test `d=1914..1941`. Strategy Statistics chạy theo scope store/department, tự grid-search alpha/beta cho từng item trên optimization, dùng cặp tốt nhất để đánh giá test, rồi ghi policy, total cost, shortage rate và average inventory cho từng item trước khi đếm strategy theo `store_id`. Strategy Visualization tổng hợp ba chỉ số này theo policy. Reference selling price mặc định là median `sell_price` của SKU trong optimization window nhưng vẫn có thể sửa tay.

Kết quả gồm daily inventory state, 3 strategy summary, grid search và các file tải xuống `daily_inventory.csv`, `strategy_summary.csv`, `grid_search_results.csv`, `run_config.json`.

Lưu ý MVP: parquet hiện không chứa calendar date nên dashboard dùng `d`; `sell_price` được lấy trực tiếp từ feature table. Khi model feature schema không tương thích, app cảnh báo và dùng seasonal-naive fallback; với artifact hiện tại, LightGBM prediction đã được kiểm thử thành công.

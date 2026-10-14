# Inventory Optimization Simulator — Luồng hiện tại

Tài liệu này mô tả phiên bản MVP đang có trong repository. Đây là tài liệu hướng dẫn sử dụng và luồng xử lý thực tế, không phải thiết kế giả định cho các module chưa tồn tại.

## 1. Mục tiêu

Ứng dụng cung cấp dashboard Streamlit để khám phá dữ liệu M5, xem forecast, mô phỏng tồn kho và benchmark ba chính sách:

1. `Historical baseline`
2. `Forecast-based`
3. `Forecast + alpha-beta`

Ứng dụng làm việc ở cấp `store_id → dept_id → item_id` và dùng dữ liệu đã xử lý trong `Data/` cùng model LightGBM trong `Models/`.

## 2. Cấu trúc repository

```text
.
├── app.py
├── src/
│   └── inventory_engine.py
├── Data/
│   └── df_detrend.parquet
├── Models/
│   └── model_{store}_{dept}.pkl
├── pipeline/
│   └── pipeline.ipynb
├── Document/
│   ├── Documentation.pdf
│   └── inventory_simulator_architecture_draft.md
└── tests/
    └── test_inventory_engine.py
```

`Data/`, `Models/` và `.venv/` là các artifact local nên được loại khỏi Git bằng `.gitignore`.

## 3. Dữ liệu và model

### Dữ liệu đầu vào cho ứng dụng

- `Data/df_detrend.parquet`: bảng feature đã xử lý từ M5, gồm demand/sales, identifier, `d` và các feature dùng khi dự báo.
- `Models/model_{store}_{dept}.pkl`: model LightGBM tương ứng với từng cặp store và department.

Ứng dụng đọc parquet theo chế độ tiết kiệm bộ nhớ, tránh tạo bản sao toàn bộ bảng vài triệu dòng. Khi người dùng chọn một SKU, app mới lọc ra frame nhỏ để phân tích và mô phỏng.

### Pipeline train

`pipeline/pipeline.ipynb` đọc dữ liệu M5 gốc, reshape dữ liệu, tạo feature, detrend, train model và lưu artifact. Kết quả được lưu vào `Data/` và `Models/` khi chạy local; trên Kaggle được lưu dưới thư mục output của Kaggle.

## 4. Các tab trong dashboard

Mỗi tab có form và selector riêng. Người dùng chọn tuần tự Store → Department → Item rồi nhấn nút Run của chính tab đó; các tab không dùng chung một khung cấu hình bên trái.

### Data Explorer

Hiển thị sales/demand và sell price của một item đã chọn. Tab này chỉ khám phá dữ liệu, không chạy mô phỏng tồn kho.

### Forecast Explorer

Hiển thị actual demand và forecast theo ngày trong hai khoảng optimization và test, kèm các chỉ số bias/MAE trên khoảng optimization.

### Inventory Simulation

Chạy mô phỏng một item với lead time, initial inventory và cost assumptions do người dùng chọn. Kết quả gồm KPI, trạng thái tồn kho theo ngày, order event, fulfillment, lost sales và forecast tổng trong lead time.

### Strategy Comparison

Chạy ba policy trên cùng item và cùng khoảng test để so sánh total cost, orders, average inventory, lost sales và service metrics.

### Optimization

Grid-search `alpha` và `beta` trên optimization window, hiển thị ma trận chi phí và cặp tham số tốt nhất. Khoảng test không được dùng để chọn alpha/beta trong tab này.

### Strategy Statistics

Chạy theo scope Store/Department và đánh giá từng item:

- tự tìm alpha/beta trên optimization window;
- ghi nhận cost, shortage rate, average inventory của từng policy;
- ghi nhận policy được khuyến nghị và bộ alpha/beta tốt nhất;
- tổng hợp số item theo policy được khuyến nghị.

### Strategy Visualization

Hiển thị **all-item benchmark**. Với mỗi policy, các chỉ số được tính trung bình trên cùng toàn bộ tập item đã chạy Strategy Statistics, không chỉ trên những item thắng policy đó. Các chỉ số gồm total cost, shortage rate và average inventory.

## 5. Khoảng thời gian đánh giá

Ứng dụng dùng các khoảng cố định:

| Khoảng | Dải `d` | Mục đích |
|---|---:|---|
| Warm-up | 28 ngày trước cửa sổ | Tạo trạng thái ban đầu và feature lịch sử |
| Optimization | `1830..1913` | Chọn policy/alpha/beta |
| Final test | `1914..1941` | Đánh giá kết quả cuối |

Lead time là tham số người dùng nhập, mặc định 10 ngày. Forecast được tạo lại theo từng ngày; tổng forecast dùng cho policy là tổng dự báo từ ngày kế tiếp đến hết lead time.

## 6. Luồng forecast và mô phỏng

Luồng xử lý cho một item:

```text
Selected Store/Department/Item
        ↓
Load filtered feature frame
        ↓
Predict daily demand with the matching LightGBM model
        ↓
Build rolling lead-time forecast
        ↓
Calculate policy parameters (ROP, Q)
        ↓
Simulate receipts, sales, inventory and lost sales
        ↓
Aggregate cost and service metrics
```

Forecast horizon bằng lead time. Với mỗi ngày mô phỏng, app dùng forecast tổng của lead-time kế tiếp. Phần lịch sử dùng để detrend/feature được giữ ở phía trước cửa sổ mô phỏng nên việc forecast lại hằng ngày không làm thay đổi quy tắc này.

## 7. Chính sách tồn kho

### Historical baseline

Dùng demand lịch sử để ước lượng demand trung bình và độ lệch chuẩn, sau đó tính safety stock, ROP và Q.

### Forecast-based

Dùng tổng forecast trong lead time để tính nhu cầu lead time, safety stock và order quantity động.

### Forecast + alpha-beta

Dùng policy forecast-based làm nền rồi điều chỉnh:

```text
ROP = ceil(alpha × ROP_forecast)
Q   = max(1, round(beta × Q_forecast))
```

Alpha/beta được tìm tự động trên grid mặc định:

```python
alpha = [0.8, 0.9, 1.0, 1.1, 1.2]
beta = [0.5, 0.75, 1.0, 1.25, 1.5]
```

## 8. Quy tắc mô phỏng

Mỗi ngày simulator thực hiện theo thứ tự:

1. Nhận các order đã đến hạn.
2. Lấy demand thực tế của ngày hiện tại.
3. Fulfill tối đa lượng tồn kho khả dụng.
4. Ghi nhận `lost_sales = max(0, demand - fulfilled)`.
5. Cập nhật on-hand inventory và inventory position.
6. Tính ROP/Q theo policy.
7. Nếu inventory position không vượt ROP, đặt một order Q units.

Inventory position bao gồm tồn kho hiện tại và các order đang vận chuyển. Lost sales không được chuyển sang ngày kế tiếp như backorder.

## 9. Chi phí và chỉ số

Tổng chi phí mô phỏng gồm:

```text
total_cost = ordering_cost + holding_cost + stockout_cost
```

Các chỉ số chính:

- `total_cost`: tổng chi phí modeled trong khoảng đánh giá;
- `shortage_rate`: tỷ lệ demand không được đáp ứng;
- `average_inventory`: tồn kho trung bình theo ngày;
- `orders`: số lần đặt hàng;
- `lost_sales`: tổng lượng thiếu hàng.

Reference selling price mặc định được lấy từ median `sell_price` của SKU trong optimization window và vẫn có thể điều chỉnh trong app.

## 10. Chạy dự án

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

Kiểm tra code và test:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py src
.\.venv\Scripts\python.exe -m pytest -q
```

## 11. Giới hạn hiện tại

- Dữ liệu M5 không cung cấp đầy đủ inventory-on-hand thực tế; sales được dùng làm demand proxy.
- Chi phí là giả định phục vụ benchmark, không phải số liệu kế toán thực tế.
- Ứng dụng hiện tập trung vào dataset/artifact đã được pipeline chuẩn bị sẵn.
- Các file dữ liệu và model lớn không được commit vào repository.

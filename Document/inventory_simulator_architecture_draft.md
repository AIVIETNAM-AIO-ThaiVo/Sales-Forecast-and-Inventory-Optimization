# Inventory Simulator — Bản thảo và kiến trúc ứng dụng

## 1. Mục tiêu

Xây dựng một ứng dụng tương tác để backtest và so sánh các chính sách quản lý tồn kho cho danh mục **M5 HOBBIES**. Ứng dụng sử dụng LightGBM models đã train để dự báo demand và một inventory simulator riêng để đánh giá chi phí vận hành.

Mục tiêu chính:

- So sánh 3 inventory strategies trên cùng SKU/store, cùng giai đoạn và cùng giả định chi phí.
- Quan sát tồn kho, thời điểm đặt hàng, thời điểm nhận hàng, lost sales và chi phí theo ngày.
- Tìm các tham số chính sách trên giai đoạn optimization, sau đó đánh giá trên giai đoạn test chưa dùng để tối ưu.

> Lưu ý: M5 cung cấp sales quan sát được, không cung cấp đầy đủ inventory-on-hand hay lượng nhu cầu bị mất khi hết hàng. Trong simulator, sales được dùng làm proxy cho demand và giả định Lost Sales phải được ghi rõ trong báo cáo.

---

## 2. Phạm vi dữ liệu và model đã có

### 2.1 Dữ liệu

Pipeline hiện tại sử dụng:

- `sales_train_evaluation.csv`: sales theo ngày.
- `calendar.csv`: ngày, `d`, `wm_yr_wk`, event và các biến lịch.
- `sell_prices.csv`: giá bán theo `item_id`, `store_id`, `wm_yr_wk`.
- `df_detrend.parquet`: feature table hiện có, gồm `d = 700–1941`, bao gồm các cột `sold`, `y`, `trend` và features.
- `cv_stats_detrend.csv`: `best_round` và thống kê CV theo từng nhóm model.
- `name_maps.json`: ánh xạ mã số sang store/dept/category/item nếu được lưu từ pipeline.

HOBBIES có **5.650 chuỗi item × store**: 4.160 thuộc `HOBBIES_1` và 1.490 thuộc `HOBBIES_2`.

### 2.2 Cấu trúc model

Có **20 LightGBM models**, mỗi model tương ứng với một cặp `(store, dept)`, không phải một model cho từng item:

```text
models_detrend/
├── model_CA_1_HOBBIES_1.pkl
├── model_CA_1_HOBBIES_2.pkl
├── model_CA_2_HOBBIES_1.pkl
├── model_CA_2_HOBBIES_2.pkl
├── ...
└── (20 models total)
```

Tên file cụ thể phải khớp với model artifacts thực tế. Ví dụ, model `CA_1 × HOBBIES_1` dự báo các chuỗi sản phẩm thuộc nhóm store/dept đó dựa trên feature table tương ứng.

Khi inference:

1. Load các `.pkl` model.
2. Load `cv_stats_detrend.csv` và lấy `best_round` tương ứng cho từng model.
3. Dựng đúng 45 features theo pipeline training.
4. Tính `pred = max(0, model(X) + trend)`.
5. Ghép dự báo với chuỗi theo `id_str` và `d`.

`BASE_W`, cách tính `trend`, danh sách feature, encoding và phiên bản LightGBM phải tương thích với lúc train. Chỉ load pickle/joblib từ nguồn đáng tin cậy.

### 2.3 Điểm cần giữ đúng về forecast

- Pipeline forecasting hiện có thiết kế horizon 28 ngày; features của ngày `d` chỉ dùng thông tin tới `d - 28`.
- Một lần dự báo 28 ngày không tự động tạo thành forecast liên tục cho bốn tháng. Để backtest tồn kho dài hơn 28 ngày, cần tạo forecast theo rolling origins, tại mỗi origin chỉ dùng dữ liệu sẵn có vào thời điểm đó.
- `df_detrend.parquet` có chứa `sold` và `y` ở giai đoạn public/test. Các cột actual của tương lai chỉ được dùng để đo kết quả simulation, **không được dùng làm features hoặc làm thông tin cho quyết định đặt hàng trước khi ngày đó xảy ra**.
- Pipeline notes ghi nhận target encoding trong train có chứa nhãn của chính dòng train. Cần rà soát/fix leakage này nếu train lại hoặc dùng kết quả để đưa ra kết luận chính thức.
- Forecast có bias và bị nén biên độ trong một số giai đoạn; kiểm tra bias và phân phối forecast error theo `dept`/SKU trên validation trước khi dùng để tính safety stock.

---

## 3. Thiết lập backtest của inventory simulator

### 3.1 Khoảng thời gian

Inventory backtest được cấu hình riêng với split của forecasting model:

- **3 tháng đầu:** optimization/validation cho inventory policy và tham số `alpha`, `beta`.
- **Tháng thứ 4:** final test để báo cáo cost.

Cần khai báo rõ ngày bắt đầu và kết thúc của bốn tháng bằng cột ngày/`d`. Không được nhầm split 3 tháng + 1 tháng của simulator với các giai đoạn train/CV/public của forecasting model.

### 3.2 Thiết lập vận hành cơ sở

| Tham số | Giá trị cơ sở | Ghi chú |
|---|---:|---|
| Lead time `L` | 10 ngày | Hàng đặt hôm nay đến sau 10 ngày theo quy ước simulator |
| Review frequency | Hàng ngày | Kiểm tra policy mỗi ngày |
| Stockout handling | Lost Sales | Nhu cầu không đáp ứng được không chuyển sang ngày sau |
| Initial inventory | Từ trạng thái warm-up | Gồm tồn kho hiện có và đơn hàng đang vận chuyển |
| Target fill rate | 95%, tùy chọn | Ràng buộc policy trên giai đoạn optimization, không phải mặc định đạt được |

Nên dùng warm-up trước giai đoạn chấm điểm để tạo tồn kho ban đầu và pipeline orders hợp lý. Nếu không có thông tin tồn kho thật, trạng thái warm-up là giả định mô phỏng và phải được ghi lại.

---

## 4. Tham số chi phí

Giá bán tham chiếu **phải lấy từ `sell_prices.csv` của dữ liệu dự án**, ghép theo `item_id`, `store_id` và `wm_yr_wk` thông qua `calendar.csv`. Không hard-code một giá bán chung cho mọi SKU.

Để có một bộ chi phí cơ sở ổn định cho MVP, dùng median `sell_price` của từng SKU/store trong giai đoạn optimization làm giá tham chiếu. Chỉ dùng các giá có sẵn trong giai đoạn đó, không lấy giá tương lai từ tháng test để thiết lập tham số.

Ký hiệu:

- `v_ref`: median selling price trong giai đoạn optimization của SKU/store.
- `c`: unit cost giả định.
- `r_h`: annual holding-cost rate.
- `h`: daily holding cost per unit.
- `K`: fixed cost per order.
- `p`: penalty mỗi unit lost sales.

Giá trị cơ sở:

| Tham số | Công thức/giá trị |
|---|---|
| Reference selling price | `v_ref` lấy từ `sell_prices.csv` |
| Cost ratio | `r_c = 0.76` |
| Unit cost | `c = 0.76 * v_ref` |
| Annual holding rate | `r_h = 0.25` |
| Annual holding cost per unit | `h_a = r_h * c` |
| Daily holding cost per unit | `h = h_a / 365` |
| Ordering cost | `K = 5` USD/order |
| Stockout penalty | `p = v_ref - c` |

Các giá trị 76%, 25% và 5 USD là **giả định mô phỏng**, không phải chi phí nội bộ thực tế của Walmart. Giữ cùng bộ giả định cho cả ba strategy; chỉ đổi chúng trong sensitivity analysis.

### Hàm cost

```math
C_total = C_ordering + C_holding + C_stockout
```

```math
C_ordering = K * N_orders
```

```math
C_holding = h * sum_t(I_t)
```

```math
C_stockout = p * sum_t(U_t)
```

```math
C_total = K * N_orders + h * sum_t(I_t) + p * sum_t(U_t)
```

Trong đó:

- `N_orders`: số lần đặt hàng trong kỳ.
- `I_t`: on-hand inventory cuối ngày `t`.
- `D_t`: demand proxy/actual sales được dùng trong ngày `t` để chấm điểm.
- `U_t = max(0, D_t - A_t)`: lost sales, với `A_t` là lượng hàng sẵn có để bán sau khi nhận hàng đến hạn.

Đây là modeled inventory-related cost gồm ordering, holding và gross-profit loss proxy. Không gọi đây là full accounting profit/cost vì chưa hạch toán đầy đủ revenue, purchase cash flow và giá trị tồn kho cuối kỳ.

### Fill rate

```math
FillRate = 1 - sum_t(U_t) / sum_t(D_t)
```

Nếu bật constraint 95%, áp dụng cùng một ngưỡng cho mọi strategy khi chọn policy trên optimization/validation period. Fill rate khác Cycle Service Level.

---

## 5. Ba inventory strategies

### Strategy 1 — Historical-data baseline

Không dùng forecasting model. Dùng historical demand trong giai đoạn optimization để xác định policy cố định.

1. Tính `mean_daily_demand` và `std_daily_demand` từ daily sales.
2. Tính safety stock bằng công thức thống kê với giả định demand ngày độc lập và lead time cố định:

```math
SS_1 = z * sigma_d * sqrt(L)
```

Dùng `z = 1.645` nếu chọn mốc tham chiếu 95% Cycle Service Level và các giả định phân phối chuẩn phù hợp.

3. Tính ROP:

```math
ROP_1 = ceil(mean_daily_demand * L + SS_1)
```

4. Tính order quantity bằng EOQ:

```math
D_annual = mean_daily_demand * 365
```

```math
Q_1 = ceil(sqrt(2 * K * D_annual / h_a))
```

`ROP_1` và `Q_1` được cố định trong lúc test. Công thức statistical safety stock là baseline xấp xỉ; không đảm bảo fill rate 95%.

### Strategy 2 — Forecast-based policy

Dùng LightGBM forecast để tính ROP, SS và Q.

1. Tại mỗi decision origin `t`, lấy dự báo cho 10 ngày tiếp theo:

```math
Dhat_10,t = sum_{j=1}^{10}(dhat_{t+j})
```

2. Trên rolling forecasts ở validation, tính sai số nhu cầu tích lũy:

```math
E_10,t = D_10,t - Dhat_10,t
```

```math
SS_2 = max(0, quantile_0.95(E_10))
```

`SS_2` được ước lượng từ validation, không dùng tháng test.

3. Tính ROP động:

```math
ROP_2,t = ceil(Dhat_10,t + SS_2)
```

4. Ước lượng annual demand từ forecast 10 ngày hiện tại:

```math
dhat_rate_t = Dhat_10,t / 10
```

```math
Dhat_annual,t = 365 * dhat_rate_t
```

```math
Q_2,t = max(1, ceil(sqrt(2 * K * Dhat_annual,t / h_a)))
```

Cách annualize này là xấp xỉ cho MVP, không phản ánh đầy đủ seasonality theo năm. Khi đặt hàng, simulator dùng Q hiện tại.

### Strategy 3 — Forecast-based policy with alpha-beta grid search

Dùng ROP và Q nền của Strategy 2, sau đó điều chỉnh hai hệ số:

- `alpha`: hệ số điều chỉnh ROP.
- `beta`: hệ số điều chỉnh order quantity.

Công thức:

```math
ROP_3,t = ceil(alpha * ROP_2,t)
```

```math
Q_3,t = max(1, ceil(beta * Q_2,t))
```

Candidate grid khởi đầu:

```python
alpha_candidates = [0.8, 0.9, 1.0, 1.1, 1.2]
beta_candidates = [0.5, 0.75, 1.0, 1.25, 1.5]
```

Grid search thử mọi cặp `(alpha, beta)` trên giai đoạn optimization, chạy simulator, tính cost và chọn cặp có cost thấp nhất. Nếu bật fill-rate constraint, chỉ xét những cặp đáp ứng constraint trên giai đoạn optimization.

```math
(alpha_star, beta_star) = argmin C_optimization(alpha, beta)
```

Không điều chỉnh `alpha`, `beta`, SS hoặc tham số policy theo kết quả tháng test.

---

## 6. Quy tắc inventory simulator theo ngày

Mỗi ngày, xử lý theo cùng một thứ tự nhất quán:

1. Nhận các đơn hàng đến hạn sau lead time 10 ngày.
2. Quan sát demand/sales của ngày hiện tại.
3. Bán `min(demand, available_inventory)` units.
4. Ghi nhận `lost_sales = max(0, demand - available_inventory)`.
5. Cập nhật on-hand inventory và inventory position.
6. Tính ROP/Q theo strategy được chọn.
7. Nếu `inventory_position <= ROP`, đặt `Q` units và ghi ngày nhận dự kiến.
8. Lưu inventory state, orders, units fulfilled, lost sales và cost components.

Inventory position với Lost Sales:

```math
IP_t = on_hand_t + outstanding_orders_t
```

Đơn hàng đang vận chuyển đã được tính vào inventory position; không được cộng lại khi đơn hàng đến nơi.

Cần dùng một quy ước ngày nhận hàng nhất quán: đơn đặt ở ngày `t` với `L=10` được nhận ở ngày `t+10`, không phải vừa đặt đã nhận ngay.

---

## 7. Kiến trúc ứng dụng

```text
inventory-simulator/
├── app.py
├── pages/
│   ├── 1_Data_Explorer.py
│   ├── 2_Forecast_Explorer.py
│   ├── 3_Inventory_Simulator.py
│   ├── 4_Strategy_Comparison.py
│   └── 5_Optimization.py
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── forecasting/
│   │   ├── model_loader.py
│   │   ├── feature_engineering.py
│   │   └── predict.py
│   ├── inventory/
│   │   ├── simulator.py
│   │   ├── policies.py
│   │   └── cost_model.py
│   ├── optimization/
│   │   ├── candidate_grids.py
│   │   ├── grid_search.py
│   │   └── sensitivity.py
│   └── evaluation/
│       ├── forecast_metrics.py
│       └── inventory_metrics.py
├── artifacts/
│   ├── models_detrend/
│   ├── cv_stats_detrend.csv
│   └── name_maps.json
├── data/
│   └── (large files stored locally, not necessarily committed to Git)
├── tests/
│   ├── test_simulator.py
│   └── test_policies.py
├── requirements.txt
└── README.md
```

### Trách nhiệm chính

- `loader.py`: đọc M5 files, calendar và sell prices; chọn category/item/store/time range.
- `preprocessing.py`: tạo long-format demand table, ghép giá theo `wm_yr_wk`, kiểm soát missing dates và giữ đúng thứ tự thời gian.
- `model_loader.py`: load 20 LightGBM models, `best_round` và name mappings.
- `feature_engineering.py`: tạo đúng 45 features, detrending và encoding theo pipeline training.
- `predict.py`: tạo rolling-origin forecast đúng horizon và không dùng dữ liệu tương lai.
- `policies.py`: triển khai logic Strategy 1–3.
- `simulator.py`: quản lý tồn kho, đơn hàng đang vận chuyển, lead time, lost sales và trạng thái hàng ngày.
- `cost_model.py`: tính ordering, holding và stockout costs.
- `grid_search.py`: chạy grid `(alpha, beta)` cho Strategy 3 trên optimization period.
- `inventory_metrics.py`: tổng hợp total cost, fill rate, average inventory, orders và lost sales.

`simulator.py` không nên phụ thuộc trực tiếp vào LightGBM. Nó nhận demand/forecast và policy thông qua interface ổn định để có thể thay forecasting model mà không viết lại engine tồn kho.

---

## 8. Thiết kế dashboard

### Sidebar / Configuration

- Category: HOBBIES.
- Store, department và item/SKU.
- Ngày bắt đầu/kết thúc optimization và test.
- Strategy 1, 2 hoặc 3.
- Lead time (mặc định 10 ngày).
- Initial inventory/warm-up setup.
- Chi phí K, holding rate, cost ratio, stockout penalty.
- Với Strategy 3: candidate ranges của `alpha`, `beta`.
- Optional fill-rate constraint.

### Tab 1 — Data Explorer

- Daily sales và sell price.
- Mean, standard deviation và tỷ lệ ngày sales bằng 0.
- Hiển thị ranh giới optimization/test.

### Tab 2 — Forecast Explorer

- Actual sales và forecast theo ngày.
- Forecast error/bias trên validation.
- Forecast sum trong từng cửa sổ 10 ngày.
- Phân phối cumulative 10-day forecast errors để giải thích safety stock.

### Tab 3 — Inventory Simulation

- On-hand inventory theo ngày.
- Dynamic/fixed ROP theo strategy.
- Sự kiện đặt hàng và nhận hàng.
- Nhu cầu, lượng fulfilled và lost sales.
- Bảng transaction/state theo ngày.

### Tab 4 — Strategy Comparison

- Total modeled cost.
- Ordering, holding và stockout cost breakdown.
- Fill rate.
- Number of orders.
- Average inventory.
- So sánh ba strategy trên cùng khoảng thời gian.

### Tab 5 — Optimization

- Kết quả từng cặp `(alpha, beta)`.
- Heatmap validation cost.
- Best feasible `(alpha, beta)`.
- Sensitivity analysis cho K, holding rate và stockout penalty.

---

## 9. Quy trình chạy và bảo vệ test set

1. Load dữ liệu và artifacts model.
2. Chọn một item/store để kiểm tra MVP.
3. Dựng forecast theo rolling origin, tuân thủ horizon/feature pipeline của model.
4. Tạo optimization window 3 tháng và final test window tháng thứ 4.
5. Tính giá bán tham chiếu từ `sell_prices.csv` trong optimization period.
6. Ước lượng SS của Strategy 2 bằng validation forecast errors.
7. Tính policy Strategy 1 và Strategy 2.
8. Grid search `(alpha, beta)` cho Strategy 3 trên optimization period.
9. Cố định toàn bộ policy tham số.
10. Chạy từng strategy trên tháng test với cùng điều kiện khởi tạo có kiểm soát.
11. Xuất bảng KPI, daily states và chart.

Không dùng actual sales của ngày tương lai để tạo features/forecast hoặc ra quyết định tại một thời điểm trước ngày đó. Actual sales trong test chỉ được đưa vào simulator đúng ngày tương ứng để tính fulfillment, lost sales và cost.

---

## 10. Kết quả app cần xuất

- `daily_inventory_{strategy}.csv`: tồn kho, inventory position, demand, forecast, ROP, Q, order arrival, fulfilled units, lost sales và cost components theo ngày.
- `strategy_summary.csv`: tổng cost và KPI của mỗi strategy trên validation/test.
- `grid_search_results.csv`: cost và fill rate theo từng cặp `(alpha, beta)`.
- `run_config.json`: thời gian backtest, SKU/store, lead time, cost assumptions và policy parameters.
- Charts/dashboard: inventory timeline, demand vs forecast, cost breakdown, strategy comparison và alpha-beta heatmap.

---

## 11. Trạng thái và việc cần xác nhận trước khi code

- [ ] Chốt đường dẫn thực tế của M5 data, model files, `cv_stats_detrend.csv` và `name_maps.json`.
- [ ] Xác nhận cách dựng lại features hoàn toàn khớp với training, bao gồm `BASE_W`.
- [ ] Xác nhận các khoảng ngày cụ thể cho 3 tháng optimization và tháng test.
- [ ] Tạo rolling-origin forecasts đủ dài cho toàn bộ backtest; model hiện có horizon 28 ngày nên cần thiết kế forecast origins cho giai đoạn dài hơn.
- [ ] Rà soát target-encoding leakage đã ghi nhận trong pipeline notes.
- [ ] Kiểm tra bias forecast theo department/SKU để safety stock không dựa trên giả định forecast không chệch.
- [ ] Viết unit tests cho order arrival, inventory position, lost sales và cost accounting.
- [ ] Kiểm tra giá bán được ghép đúng `item_id × store_id × wm_yr_wk` và chỉ dùng thông tin có sẵn.


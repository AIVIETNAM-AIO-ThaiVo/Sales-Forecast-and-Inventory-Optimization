# Inventory Optimization Simulator

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)
![License](https://img.shields.io/badge/status-MVP-orange)

> Dashboard mô phỏng và đánh giá các chính sách tồn kho trên dữ liệu M5 Forecasting Accuracy.

## 📚 Project Documentation

Đọc tài liệu dự án trước khi sử dụng:

- **[Documentation đầy đủ (PDF)](Document/Documentation.pdf)** — tài liệu tổng quan và hướng dẫn chi tiết.
- **[Luồng hoạt động hiện tại](Document/inventory_simulator_architecture_draft.md)** — mô tả đúng các tab, pipeline dữ liệu, forecast và simulator đang có.

Các tài liệu trên được đặt ngay trong repository để người đọc có thể mở trực tiếp từ README.

## ✨ Highlights

- Dashboard Streamlit gồm 7 khu vực phân tích độc lập.
- So sánh Historical Baseline, Forecast-based và Forecast + Alpha-Beta.
- Forecast theo đúng lead time và mô phỏng tồn kho theo từng ngày.
- Tự tìm alpha/beta phù hợp cho từng item trong bước Strategy Statistics.
- Tính ROP bằng `ceil` và Q bằng `round`.
- Benchmark toàn bộ item theo tổng chi phí, tỷ lệ thiếu hàng và tồn kho trung bình.
- Pipeline notebook để xử lý dữ liệu M5, train LightGBM và xuất artifact cho ứng dụng.

## 🔎 Overview

Inventory Optimization Simulator là một MVP giúp phân tích nhu cầu, dự báo ngắn hạn và đánh giá chính sách bổ sung hàng ở cấp Store–Department–Item.

Ứng dụng sử dụng dữ liệu đã xử lý trong `Data/` và các model LightGBM trong `Models/`. Người dùng có thể khám phá dữ liệu, xem forecast, chạy mô phỏng, tối ưu tham số và so sánh benchmark giữa các policy.

Các cửa sổ đánh giá mặc định:

- Optimization: `d=1830..1913`
- Final test: `d=1914..1941`
- Warm-up: 28 ngày

### 👤 Author

Dự án cá nhân phục vụ nghiên cứu và thử nghiệm inventory optimization trên dữ liệu M5.

## 🚀 Usage

Khởi động dashboard:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Sau khi mở ứng dụng, chọn Store → Department → Item trong từng tab rồi nhấn nút Run tương ứng. Các tab gồm:

- Data Explorer
- Forecast Explorer
- Inventory Simulation
- Strategy Comparison
- Optimization
- Strategy Statistics
- Strategy Visualization

### Giao diện từng tab

#### Data Explorer

![Data Explorer](<picture/Data Explorer.png>)

#### Forecast Explorer

![Forecast Explorer](<picture/Forecast.png>)

#### Inventory Simulation

![Inventory Simulation](<picture/Inventory Simulator.png>)

#### Strategy Comparison

![Strategy Comparison](<picture/Strategy Comparision.png>)

#### Strategy Visualization

![Strategy Visualization](<picture/Strategy Visualization.png>)

Kết quả có thể tải xuống dưới dạng `daily_inventory.csv`, `strategy_summary.csv`, `grid_search_results.csv` và `run_config.json`.

## 📦 Installation

Yêu cầu Python 3.10 trở lên. Khuyến nghị sử dụng virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Đặt artifact đầu vào như sau:

```text
Data/
└── df_detrend.parquet

Models/
├── model_CA_1.pkl
├── model_CA_2.pkl
└── ...
```

Các thư mục `Data/`, `Models/` và `.venv/` được đưa vào `.gitignore` vì chứa dữ liệu lớn, model và môi trường local.

### Training pipeline

Notebook xử lý và train model nằm tại [pipeline/pipeline.ipynb](pipeline/pipeline.ipynb). Pipeline hỗ trợ:

1. Đọc dữ liệu M5.
2. Reshape và tạo feature theo item.
3. Detrend, tạo lag/rolling features và encoding.
4. Train các model LightGBM theo Store–Department.
5. Lưu dữ liệu đã xử lý vào `Data/` và model vào `Models/`.

Trên local, dữ liệu M5 mặc định được tìm tại:

```text
raw_data/m5-forecasting-accuracy/
```

Có thể thay đổi đường dẫn bằng biến môi trường `M5_INPUT_DIR`.

## 🤝 Feedback and Contributing

Nếu phát hiện lỗi hoặc có đề xuất cải tiến, hãy mở một Issue trong repository. Các đóng góp liên quan đến giao diện, hiệu năng, forecasting hoặc inventory policy đều được hoan nghênh.

Trước khi tạo pull request, hãy kiểm tra cú pháp và chạy test:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py src
.\.venv\Scripts\python.exe -m pytest -q
```

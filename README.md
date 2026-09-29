# Tái lập nghiên cứu chấm điểm tín dụng

Dự án so sánh ba mô hình dự báo vỡ nợ thẻ tín dụng: **Logistic WoE**, **Random Forest** và **XGBoost**. Pipeline chính dùng dữ liệu UCI gốc và đúng danh sách khách hàng đã chia trong bài báo. CSV sạch do nhóm cung cấp vẫn được giữ riêng để đối chiếu, không bị ghi đè.

## Chạy toàn bộ phân tích

### Windows

1. Cài Python **3.13**.
2. Mở PowerShell ngay trong thư mục `Credit_scoring`.
3. Chạy lệnh:

```powershell
py -3.13 run_project.py
```

Lần đầu chạy, chương trình tự tạo môi trường `.venv`, cài các thư viện đã khóa phiên bản rồi khởi động phân tích. Cần Internet để tải thư viện lần đầu. Lần chạy đầy đủ gồm kiểm định chéo lồng nhau, hiệu chỉnh xác suất, bootstrap, SHAP và robustness nên có thể mất nhiều thời gian.

### macOS hoặc Linux

```bash
python3.13 run_project.py
```

Nếu lệnh `py` hoặc `python3.13` không chạy, hãy cài Python 3.13 rồi mở terminal mới tại thư mục dự án. Khi một bước lỗi, chương trình dừng và hiển thị thông báo trong terminal.

Kết quả được tạo trong `results/`, `figures/`, `models/` và `data/generated/`. Chạy lại sẽ thay các kết quả cùng tên; hãy sao chép kết quả cũ sang nơi khác nếu cần lưu.

## Kiểm tra nhanh trước khi chạy toàn bộ

Smoke test giúp người dùng kiểm tra dữ liệu và ba mô hình có chạy được không mà không phải chờ phân tích đầy đủ. Test lấy mẫu ngẫu nhiên có phân tầng **20% dữ liệu** theo nhãn, chia mẫu đó thành train/test 80/20, rồi fit ba mô hình với cấu hình cơ sở. Test **không** chạy tuning tham số, bootstrap, SHAP hoặc robustness. AUC của mẫu nhỏ chỉ dùng để kiểm tra phần mềm, không dùng để đối chiếu với kết quả bài báo.

Mở PowerShell tại thư mục dự án và chạy lần lượt:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tests\smoke_models.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Trên macOS/Linux, thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python`. Có thể chạy smoke test nhiều lần; cùng cấu hình sẽ cho cùng mẫu và seed.

## Thiết lập tái lập trong pipeline chính

- **Dữ liệu:** `data/reference/default of credit card clients.xls` (30.000 hồ sơ, 6.636 nhãn vỡ nợ). Pipeline giữ nguyên các mã gốc của học vấn, hôn nhân và trạng thái trả nợ.
- **Nhãn:** `default payment next month`, bằng 1 nếu khách hàng vỡ nợ trong tháng kế tiếp, bằng 0 nếu không.
- **Đầu vào mô hình:** hạn mức tín dụng, tuổi, học vấn, tình trạng hôn nhân, lịch sử trả nợ, hóa đơn và khoản thanh toán. `ID` và `SEX` không được dùng làm biến dự báo. `SEX` được giữ để phân tích nhóm; chữ ký nhóm hồ sơ dùng toàn bộ đặc điểm, bao gồm giới tính, nhưng loại `ID` và nhãn.
- **Chia mẫu cố định:** bài báo dùng `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260927)` và lấy fold đầu tiên. `data/splits/article_split_membership.csv` là nguồn chính thức cho nhãn development/test; `article_test_ids.csv` là danh sách đối chiếu. Có 23.999 hồ sơ development (5.309 ca vỡ nợ) và 6.001 hồ sơ test (1.327 ca vỡ nợ). Chương trình kiểm tra ID, số lượng, nhãn và hồ sơ trùng trước khi chạy. Pipeline dùng membership đã lưu để không phụ thuộc khác biệt phiên bản thư viện khi tạo fold.
- **Seed mô hình và CV:** `20260927`; các bước CV dùng seed dẫn xuất như đã ghi trong `credit_scoring/analysis.py`.
- **Chọn mô hình:** nested group CV gồm 5 fold ngoài và 3 fold trong; cấu hình cuối cùng được chọn bằng group CV 5 fold chỉ trên development. Lưới tham số và các thiết lập mô hình nằm trong `credit_scoring/analysis.py`.
- **Kết quả tham chiếu từ dự đoán bài báo:** AUC Logistic WoE **0,7703**, Random Forest **0,7827**, XGBoost **0,7840**. Cấu hình Random Forest được chọn trên split bài báo là depth 10 và leaf tối thiểu 10; cấu hình được pipeline ghi lại trong `results/selected_parameters.json` sau khi chạy.

`data/splits/article_test_predictions.csv` lưu dự đoán tham chiếu của bài báo để dùng trong các phép đối chiếu tùy chọn. Pipeline không dùng các dự đoán này để fit hay chọn mô hình.

## Thư mục và tệp kết quả

```text
data/cleaned/       CSV sạch của nhóm, giữ riêng để phân tích độ nhạy
data/generated/     Split được xuất, dự đoán OOF và dự đoán test
data/reference/     Workbook UCI gốc và tài liệu tham khảo
data/splits/        Split bài báo, danh sách ID và dự đoán tham chiếu
credit_scoring/     Mã phân tích, robustness bổ sung và finalize
experiments/        Các phép đối chiếu tập trung, không chạy full pipeline
tests/              Contract tests và smoke test nhanh
results/            Bảng, metric, integrity checks và phiên bản phần mềm
figures/            Hình được tạo khi chạy
models/             Mô hình đã fit
```

Sau khi pipeline hoàn tất, mở `results/integrity_checks.json`. Chỉ sử dụng bảng/hình khi tất cả giá trị kiểm tra đều là `true`. Dữ liệu nguồn `.xls` không bị sửa.

## Phép đối chiếu tùy chọn

Sau khi đã có môi trường và kết quả full pipeline, các lệnh sau chạy những kiểm tra hẹp hơn, không chạy bootstrap, SHAP hoặc robustness:

```powershell
.\.venv\Scripts\python.exe experiments\compare_source_encoding.py
.\.venv\Scripts\python.exe experiments\rerun_article_split.py
.\.venv\Scripts\python.exe experiments\retune_article_random_forest.py
```

Các kết quả chẩn đoán được ghi vào `results/`. Không chọn seed theo AUC để làm cho con số gần bài báo hơn.

## Cách diễn giải

Dữ liệu phản ánh khách hàng thẻ tín dụng tại Đài Loan, không đại diện cho danh mục của một ngân hàng hiện tại. Utility là kịch bản chi phí chuẩn hóa, không phải lợi nhuận ngân hàng quan sát được. SHAP mô tả dự đoán của mô hình, không phải tác động nhân quả.

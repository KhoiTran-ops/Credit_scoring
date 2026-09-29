# Credit scoring research project

Project này chạy lại toàn bộ phân tích từ `data/cleaned/Du_lieu_sach_Chu_de_1.csv` do nhóm cung cấp. File `.xls` gốc trong `data/reference/` chỉ được giữ để đối chiếu, **không** được dùng làm đầu vào và code không làm sạch lại các giá trị của CSV. Trong lúc đọc, code ánh xạ 25 tên cột của CSV sang tên biến UCI mà các mô hình hiện có sử dụng (ví dụ `repayment_status_sep` → `PAY_0`).

## Cấu trúc project

```text
run_project.py                 File chạy chính
credit_scoring/               Các module phân tích mô hình
data/cleaned/                 CSV sạch làm đầu vào duy nhất
data/reference/               Dữ liệu UCI gốc và thống kê mô tả để đối chiếu
data/generated/               Dự báo và thông tin chia mẫu được tạo khi chạy
results/, figures/, models/   Bảng kết quả, hình và mô hình được tạo khi chạy
```

## Chạy trên máy khác

Cần Python **3.13** (đã kiểm với 3.13.5), kết nối Internet trong lần cài thư viện đầu tiên và đủ dung lượng cho môi trường cùng các kết quả. Sao chép cả thư mục project, gồm `credit_scoring/`, `data/cleaned/`, `run_project.py` và `requirements.txt`. Không sao chép `.venv` từ máy cũ.

Mở terminal tại thư mục này và chạy **một lệnh**:

```powershell
py -3.13 run_project.py
```

Trên macOS/Linux, dùng `python3.13 run_project.py`. Nếu Windows không nhận lệnh `py`, hãy cài Python 3.13 hoặc gọi trực tiếp đường dẫn đến `python.exe` phiên bản 3.13. Script tự tạo `.venv` trong project nếu chưa có, cài đúng phiên bản thư viện trong `requirements.txt`, rồi chạy lần lượt:

1. `credit_scoring.analysis`: chia tập theo nhóm hồ sơ, huấn luyện và đánh giá ba mô hình, bootstrap, SHAP, kiểm tra độ vững R1–R9.
2. `credit_scoring.robustness_supplement`: hiệu chỉnh lại mô hình có trọng số, nhóm tuổi thay thế, khoảng tin cậy giá trị quyết định.
3. `credit_scoring.finalize_outputs`: kiểm tra tính toàn vẹn và tạo các bảng/hình tổng hợp cuối.

Các kết quả được ghi vào `results/`, `figures/`, `data/generated/` và `models/`. Lần chạy lại sẽ ghi đè tệp kết quả cùng tên; giữ bản sao nếu muốn lưu một lần chạy cũ. `results/software_versions.json` ghi phiên bản thực tế đã dùng. Vì có kiểm định chéo lồng nhau và hàng trăm lần huấn luyện bootstrap, lần chạy đầy đủ có thể mất nhiều thời gian.

## Kiểm tra nhanh

Sau khi môi trường đã được tạo, chạy:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Trên macOS/Linux thay đường dẫn Python bằng `.venv/bin/python`. Một lần chạy thành công phải kết thúc với `results/integrity_checks.json` chỉ chứa giá trị `true`. Nếu script dừng giữa chừng, xem lỗi hiển thị trong terminal; các tệp kết quả có thể chưa đầy đủ và không nên dùng để báo cáo.

## Tái lập và giới hạn so sánh

CSV sạch là nguồn duy nhất. Code không dùng CSV cache hoặc `.xls` gốc khi chạy. Dữ liệu sạch đã gộp một số mã học vấn/hôn nhân so với `.xls` gốc, nên kết quả có thể khác bài demo từng chạy từ dữ liệu gốc. Ngoài ra, các thư viện được khóa phiên bản nhưng sai khác số học rất nhỏ giữa hệ điều hành/phần cứng vẫn có thể xảy ra. Không diễn giải lợi ích/tổn thất chuẩn hóa trong code là lợi nhuận ngân hàng quan sát được.

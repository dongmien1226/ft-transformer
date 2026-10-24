"""
config.py - Nguon chan ly duy nhat cho toan bo thuc nghiem.

Nguyên tắc: MỌI siêu tham số nằm ở đây. Không hard-code trong script.
Dict cấu hình được truyền thẳng vào model_config_table() của report_export.py
để sinh Bảng 3.6 / 3.7, nhờ đó bảng trong báo cáo không thể lệch khỏi
cấu hình đã thực sự sinh ra số liệu.

Đặc tả bám theo Chương 2, Bảng 2.7 (kiến trúc) và Bảng 2.8 (huấn luyện).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# config.py được MỌI script khác import đầu tiên (from flow_based import config as C) - đây
# là chỗ tự nhiên duy nhất để vá một lỗi có sẵn trong toàn bộ pipeline: trên
# Windows, khi stdout bị pipe (qua `tail`, `Select-Object`, ghi log file...)
# Python thường không mặc định UTF-8, mà toàn bộ script đều in tiếng Việt có
# dấu -> UnicodeEncodeError làm chết cả script giữa chừng. Colab (Linux) luôn
# UTF-8 nên không gặp, nhưng vá ở đây để chạy an toàn trên mọi môi trường.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# ĐƯỜNG DẪN - tự nhận biết Colab hay máy cá nhân
# ---------------------------------------------------------------------------
ON_COLAB = os.path.isdir("/content")

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = Path(
    os.environ.get("IDS_RAW_DIR", PROJECT_DIR / "data" / "raw" / "MachineLearningCVE")
)
PROC_DIR = Path(os.environ.get("IDS_PROC_DIR", PROJECT_DIR / "data" / "processed"))
RESULTS_DIR = Path(os.environ.get("IDS_RESULTS_DIR", PROJECT_DIR / "ketqua"))

# Thư mục lưu trữ lâu dài trên Drive. Session Colab bị xóa sạch khi ngắt kết nối,
# nên mọi thứ cần giữ lại phải được copy về đây sau MỖI cấu hình.
DRIVE_DIR = Path(os.environ.get("IDS_DRIVE_DIR", "/content/drive/MyDrive/DATN"))

# ---------------------------------------------------------------------------
# TÊN 8 FILE CSV CỦA CIC-IDS2017 (bộ MachineLearningCVE)
# ---------------------------------------------------------------------------
# File DUY NHẤT chứa toàn bộ mẫu tấn công web - bắt buộc phải có.
WEB_ATTACK_FILE = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"

EXPECTED_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    WEB_ATTACK_FILE,
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# ---------------------------------------------------------------------------
# PHẠM VI LỚP - chỉ tấn công Web, đúng tên đề tài
# ---------------------------------------------------------------------------
# Đề tài: "Phát hiện các cuộc tấn công Web dựa trên phân tích dữ liệu mạng".
# DoS/DDoS/PortScan/Bot/Heartbleed là tấn công tầng mạng-vận tải, KHÔNG thuộc
# nhóm tấn công ứng dụng web theo phân loại OWASP -> bị loại khỏi phạm vi.
WEB_ONLY = True

CLASS_NAMES = ["BENIGN", "Web-BruteForce", "Web-XSS", "Web-SQLi"]
WEB_ATTACK_CLASSES = ["Web-BruteForce", "Web-XSS", "Web-SQLi"]

# Nhãn nhị phân: 0 = BENIGN, 1 = Web Attack
BINARY_NAMES = ["BENIGN", "Web-Attack"]

# Số mẫu kỳ vọng của từng lớp web trong CIC-IDS2017. Dùng để ASSERT CỨNG sau
# khi map nhãn. Lần chạy trước, nhãn bị hỏng mã hóa nên không khớp bảng ánh xạ
# và chỉ số web_macro_recall bằng NaN suốt 5 lần chạy mà không ai phát hiện.
EXPECTED_WEB_COUNTS = {
    "Web-BruteForce": 1507,
    "Web-XSS": 652,
    "Web-SQLi": 21,
}

# Nguồn lấy lưu lượng bình thường:
#   "same_file"  - benign CÙNG buổi sáng thứ Năm với các cuộc tấn công (~168K mẫu).
#                  Sạch về mặt khoa học: loại bỏ yếu tố gây nhiễu "mô hình học
#                  ngày nào" thay vì học hành vi tấn công. Chỉ cần 1 file CSV.
#   "all_files"  - toàn bộ ~1,65 triệu benign của cả 8 file. Tỷ số mất cân bằng
#                  lên tới ~760:1, chậm hơn nhiều, và có nguy cơ rò rỉ theo ngày.
BENIGN_SOURCE = "same_file"

# ---------------------------------------------------------------------------
# TIỀN XỬ LÝ - Chương 2 mục 2.3, Bảng 2.6 (sáu biện pháp chống rò rỉ)
# ---------------------------------------------------------------------------
# Cột định danh và cột thời gian: trong testbed CIC-IDS2017 kẻ tấn công dùng IP
# cố định và tấn công diễn ra trong khung giờ cố định. Giữ lại thì mô hình chỉ
# học thuộc địa chỉ/thời điểm và cho accuracy ~99,9% hoàn toàn vô nghĩa.
LEAKY_COLS = [
    "Flow ID",
    "Source IP",
    "Src IP",
    "Destination IP",
    "Dst IP",
    "Source Port",
    "Src Port",
    "Timestamp",
    "Protocol",
]

# Destination Port cực kỳ "mạnh" nhưng phản ánh đặc thù testbed hơn là bản chất
# tấn công. Bỏ đi -> kết quả thấp hơn nhưng trung thực và dễ bảo vệ hơn.
DROP_DEST_PORT = True

DEDUPLICATE = True  # Khử trùng lặp TRƯỚC khi chia tập (Bảng 2.6, biện pháp 1)
DROP_CONSTANT_COLS = True  # Bỏ cột phương sai = 0 (CIC-IDS2017 có ~8 cột toàn số 0)

TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Chương 2 mục 2.3.3a đã chốt phép biến đổi phân vị: đặc trưng luồng có đuôi rất
# dài (Flow Duration, Bytes/s...) nên chuẩn hóa theo trung bình không khống chế
# được giá trị ngoại lệ.
SCALER = "quantile"  # 'quantile' | 'standard' | 'robust'

SUBSAMPLE = None  # Đặt số nguyên để debug nhanh. None = dùng toàn bộ.

# ---------------------------------------------------------------------------
# TÁI LẬP
# ---------------------------------------------------------------------------
SEED = 42
SEEDS = [
    42,
    1337,
    2024,
]  # Cùng danh sách cho CẢ 5 mô hình; SEED=42 giữ phép chia cố định.

# ---------------------------------------------------------------------------
# MÔ HÌNH TRANSFORMER ĐỀ XUẤT - Chương 2, Bảng 2.7
# ---------------------------------------------------------------------------
MODEL = dict(
    d_model=64,  # Đủ biểu diễn từng đặc trưng mà vẫn giữ mô hình gọn
    n_layers=3,  # Dữ liệu bảng không đòi hỏi độ sâu như dữ liệu ngôn ngữ
    n_heads=8,  # d_k = d_model / n_heads = 8
    d_ff=128,  # Bằng 2 x d_model, hạn chế số tham số
    dropout=0.1,
    activation="gelu",  # Trơn hơn ReLU, phù hợp kiến trúc Transformer
    norm_first=True,  # Pre-LN: ổn định huấn luyện, không cần giai đoạn khởi động
)

# Mạng nơ-ron truyền thẳng làm mô hình cơ sở. Vai trò đặc biệt (Chương 3 mục
# 3.2.3): tách phần đóng góp của HỌC SÂU NÓI CHUNG khỏi phần đóng góp của RIÊNG
# cơ chế tự chú ý. Vì vậy MLP dùng CHUNG vòng lặp huấn luyện với Transformer.
MLP = dict(
    hidden_dims=[256, 128],
    dropout=0.1,
    activation="gelu",
)

# ---------------------------------------------------------------------------
# HUẤN LUYỆN - Chương 2, Bảng 2.8
# ---------------------------------------------------------------------------
TRAIN = dict(
    optimizer="AdamW",  # Tách suy giảm trọng số khỏi bước cập nhật gradient
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=512,  # Đủ lớn để mỗi lô chứa được vài mẫu của lớp thưa
    max_epochs=100,  # Giới hạn trên; thực tế dừng sớm hơn
    scheduler="ReduceLROnPlateau",
    lr_factor=0.5,  # Giảm một nửa khi kiểm định không cải thiện
    lr_patience=5,
    # Early stopping đặt lớn hơn lr_patience để cho phép 2-3 lần giảm lr
    # phát huy tác dụng trước khi dừng hẳn.
    early_stop_patience=15,
    monitor="val_macro_f1",  # KHÔNG dùng accuracy để dừng sớm
    # Tập val chỉ có 98 mẫu Web-XSS, 3 mẫu Web-SQLi -> val_macro_f1 dao động
    # rất mạnh giữa các epoch liên tiếp (quan sát thực tế: 0.32 -> 0.73 -> 0.66
    # chỉ trong vài epoch ở lần chạy run_20260830_161848). Làm mượt bằng trung
    # bình trượt K epoch gần nhất trước khi quyết định "có cải thiện hay
    # không", để giảm rủi ro epoch "tốt nhất" chỉ là một đỉnh nhiễu ngẫu
    # nhiên. early_stop_smoothing=1 tắt hoàn toàn tính năng này (hành vi cũ).
    early_stop_smoothing=3,
    grad_clip=1.0,
    num_workers=2,
    # Gán trọng số lớp theo Chương 2 mục 2.5.1; "none" để tắt trọng số.
    imbalance="class_weight",
)

# ---------------------------------------------------------------------------
# MÔ HÌNH CƠ SỞ - Chương 3, Bảng 3.6
# ---------------------------------------------------------------------------
LOGREG_PARAMS = dict(
    solver="saga",
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEED,
)

RF_PARAMS = dict(
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=SEED,
)

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=SEED,
)

# ---------------------------------------------------------------------------
# ÁNH XẠ NHÓM cho model_config_table() -> cột "Nhóm" của Bảng 3.7
# ---------------------------------------------------------------------------
GROUP_MAP = {
    "Số token đầu vào (F)": "Biểu diễn đầu vào",
    "Cơ chế embedding đặc trưng": "Biểu diễn đầu vào",
    "Vector định danh cột": "Biểu diễn đầu vào",
    "Positional encoding": "Biểu diễn đầu vào",
    "Số khối Transformer Encoder (N)": "Encoder",
    "Số đầu chú ý (h)": "Encoder",
    "Kích thước embedding (d_model)": "Encoder",
    "Kích thước lớp feed-forward (d_ff)": "Encoder",
    "Tỷ lệ dropout": "Encoder",
    "Hàm kích hoạt / vị trí chuẩn hóa": "Encoder",
    "Cơ chế pooling": "Đầu ra",
    "Cấu trúc classification head": "Đầu ra",
    "Hàm mất mát": "Huấn luyện",
    "Thuật toán tối ưu": "Huấn luyện",
    "Learning rate ban đầu / weight decay": "Huấn luyện",
    "Learning-rate scheduler": "Huấn luyện",
    "Batch size": "Huấn luyện",
    "Số epoch tối đa": "Huấn luyện",
    "Điều kiện dừng sớm (chỉ số / patience)": "Huấn luyện",
    "Làm mượt chỉ số dừng sớm (cửa sổ)": "Huấn luyện",
    "Tổng số tham số khả huấn luyện": "Quy mô",
}


def bang_3_7(
    n_features: int,
    n_classes: int,
    n_params: int,
    model_cfg: dict | None = None,
    train_cfg: dict | None = None,
) -> dict:
    """Dựng đúng dict cho Bảng 3.7 từ CẤU HÌNH ĐANG CHẠY.

    Không gõ lại giá trị nào vào Word - truyền kết quả hàm này vào
    model_config_table(bang_3_7(...), GROUP_MAP).
    """
    m = model_cfg if model_cfg is not None else MODEL
    t = train_cfg if train_cfg is not None else TRAIN
    return {
        "Số token đầu vào (F)": f"{n_features} (+1 token [CLS])",
        "Cơ chế embedding đặc trưng": "Token hóa đặc trưng: e_j = x_j * w_j + b_j",
        "Vector định danh cột": "Có - w_j, b_j riêng cho từng cột",
        "Positional encoding": "Không sử dụng (thứ tự cột không mang ý nghĩa)",
        "Số khối Transformer Encoder (N)": m["n_layers"],
        "Số đầu chú ý (h)": f"{m['n_heads']} (d_k = {m['d_model'] // m['n_heads']})",
        "Kích thước embedding (d_model)": m["d_model"],
        "Kích thước lớp feed-forward (d_ff)": m["d_ff"],
        "Tỷ lệ dropout": m["dropout"],
        "Hàm kích hoạt / vị trí chuẩn hóa": f"{m['activation'].upper()} / "
        f"{'Pre-LN (chuẩn hóa trước)' if m['norm_first'] else 'Post-LN'}",
        "Cơ chế pooling": "Token [CLS]",
        "Cấu trúc classification head": f"Linear(GELU(LayerNorm(z_cls))) -> {n_classes} lớp",
        "Hàm mất mát": "Entropy chéo có trọng số lớp, w_c = N_train / (C * n_c)"
        if t["imbalance"] == "class_weight"
        else "Entropy chéo",
        "Thuật toán tối ưu": t["optimizer"],
        "Learning rate ban đầu / weight decay": f"{t['lr']} / {t['weight_decay']}",
        "Learning-rate scheduler": f"{t['scheduler']} (factor {t['lr_factor']}, patience {t['lr_patience']})",
        "Batch size": t["batch_size"],
        "Số epoch tối đa": t["max_epochs"],
        "Điều kiện dừng sớm (chỉ số / patience)": f"{t['monitor']} / {t['early_stop_patience']} epoch",
        "Làm mượt chỉ số dừng sớm (cửa sổ)": f"trung bình trượt {t.get('early_stop_smoothing', 1)} epoch"
        if t.get("early_stop_smoothing", 1) > 1
        else "không (giá trị thô)",
        "Tổng số tham số khả huấn luyện": f"{n_params:,}",
    }


def bang_3_6(train_cfg: dict | None = None, seeds: list[int] | None = None) -> dict:
    """Bảng 3.6 - siêu tham số của bốn mô hình cơ sở, lấy từ chính dict đang chạy."""
    t = train_cfg if train_cfg is not None else TRAIN
    seeds = SEEDS if seeds is None else seeds

    def describe(params):
        return ", ".join(
            f"{k}={seeds if k == 'random_state' else v}" for k, v in params.items()
        )

    return {
        "Logistic Regression": describe(LOGREG_PARAMS),
        "Random Forest": describe(RF_PARAMS),
        "XGBoost": describe(XGB_PARAMS),
        "MLP": (
            f"hidden_dims={MLP['hidden_dims']}, dropout={MLP['dropout']}, "
            f"activation={MLP['activation']}, optimizer={t['optimizer']}, "
            f"lr={t['lr']}, batch_size={t['batch_size']}, "
            f"max_epochs={t['max_epochs']}, seeds={seeds}"
        ),
    }


DATASETS = {
    "cicids2017": {
        "name": "CIC-IDS2017 (MachineLearningCVE)",
        "raw": "MachineLearningCVE",
        "processed": "",
        "results": "",
        "web_counts": {"Web-BruteForce": 1507, "Web-XSS": 652, "Web-SQLi": 21},
    },
    "cse_cicids2018": {
        "name": "CSE-CIC-IDS2018 (web-attack day 23 February)",
        "attack_day": "2018-02-23",
        "raw": "cse_cicids2018",
        "processed": "cse_cicids2018/2018-02-23",
        "results": "cse_cicids2018/2018-02-23",
        "web_counts": {"Web-BruteForce": 362, "Web-XSS": 151, "Web-SQLi": 53},
    },
}


def select_dataset(dataset: str) -> None:
    """Chọn dữ liệu và đường dẫn; không thay đổi kiến trúc hoặc siêu tham số."""
    global \
        DATASET_ID, \
        DATASET_NAME, \
        ATTACK_DAY, \
        RAW_DIR, \
        PROC_DIR, \
        RESULTS_DIR, \
        EXPECTED_WEB_COUNTS
    if dataset not in DATASETS:
        raise ValueError(f"Dataset không hỗ trợ: {dataset!r}")
    profile = DATASETS[dataset]
    DATASET_ID, DATASET_NAME = dataset, profile["name"]
    ATTACK_DAY = profile.get("attack_day")
    RAW_DIR = Path(
        os.environ.get("IDS_RAW_DIR", PROJECT_DIR / "data" / "raw" / profile["raw"])
    )
    PROC_DIR = Path(
        os.environ.get(
            "IDS_PROC_DIR", PROJECT_DIR / "data" / "processed" / profile["processed"]
        )
    )
    RESULTS_DIR = Path(
        os.environ.get("IDS_RESULTS_DIR", PROJECT_DIR / "ketqua" / profile["results"])
    )
    EXPECTED_WEB_COUNTS = dict(profile["web_counts"])


select_dataset(os.environ.get("IDS_DATASET", "cicids2017"))


if __name__ == "__main__":
    print(f"Chạy trên Colab      : {ON_COLAB}")
    print(f"Thư mục dữ liệu thô  : {RAW_DIR}")
    print(f"Thư mục đã tiền xử lý: {PROC_DIR}")
    print(f"Thư mục kết quả      : {RESULTS_DIR}")
    print(f"Phạm vi lớp          : {CLASS_NAMES}")
    print(f"Nguồn benign         : {BENIGN_SOURCE}")
    print(f"Seeds                : {SEEDS}")

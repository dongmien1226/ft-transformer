"""
verify_data.py - Kiểm tra tính toàn vẹn của bộ dữ liệu TRƯỚC khi tiền xử lý.

Chạy:  python -m flow_based.data.verify_data

VÌ SAO CẦN BƯỚC NÀY
===================
Lần chạy trước, lệnh giải nén báo "zipfile read error" nhưng script vẫn chạy
tiếp với 5/8 file CSV. Kết quả: 1,92 triệu dòng thay vì 2,83 triệu, và không ai
nhận ra cho tới khi đối chiếu lại toàn bộ log. Bước kiểm tra này chặn đúng
tình huống đó.

Với phạm vi "chỉ tấn công Web" (config.WEB_ONLY = True), file BẮT BUỘC duy nhất
là Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv - nơi chứa toàn bộ
2.180 mẫu tấn công web của CIC-IDS2017. Các file còn lại chỉ cần khi
config.BENIGN_SOURCE = "all_files".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from flow_based import config as C
from flow_based.data.labels import distribution


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def check_files() -> tuple[list[str], list[str]]:
    """Đối chiếu các file CSV thực có với danh sách 8 file kỳ vọng."""
    print("=" * 78)
    print("KIỂM TRA TÍNH TOÀN VẸN BỘ DỮ LIỆU CIC-IDS2017")
    print("=" * 78)
    print(f"\nThư mục: {C.RAW_DIR}\n")

    if not C.RAW_DIR.is_dir():
        sys.exit(
            f"[LỖI] Không tìm thấy thư mục {C.RAW_DIR}\n"
            f"       Tải MachineLearningCSV.zip từ\n"
            f"       https://www.unb.ca/cic/datasets/ids-2017.html\n"
            f"       rồi giải nén 8 file CSV vào thư mục trên."
        )

    present, missing = [], []
    for name in C.EXPECTED_FILES:
        p = C.RAW_DIR / name
        if p.is_file():
            present.append(name)
            print(f"  [CÓ    ] {name:<58} {_human(p.stat().st_size)}")
        else:
            missing.append(name)
            print(f"  [THIẾU ] {name}")

    extra = sorted(
        f.name for f in C.RAW_DIR.glob("*.csv") if f.name not in C.EXPECTED_FILES
    )
    for name in extra:
        print(f"  [LẠ    ] {name}  <- không thuộc bộ MachineLearningCVE")

    print(f"\n  Tổng: {len(present)}/8 file kỳ vọng có mặt.")
    return present, missing


def check_required(missing: list[str]) -> None:
    """Dừng cứng nếu thiếu file bắt buộc cho cấu hình hiện tại."""
    print("\n" + "-" * 78)
    print(
        f"Phạm vi đang cấu hình: WEB_ONLY={C.WEB_ONLY}, "
        f"BENIGN_SOURCE={C.BENIGN_SOURCE!r}"
    )
    print("-" * 78)

    if C.WEB_ATTACK_FILE in missing:
        sys.exit(
            f"\n[LỖI NGHIÊM TRỌNG] Thiếu file BẮT BUỘC:\n"
            f"    {C.WEB_ATTACK_FILE}\n\n"
            f"Đây là file DUY NHẤT chứa mẫu tấn công web của CIC-IDS2017.\n"
            f"Không có nó thì không thể chạy thực nghiệm này."
        )

    if C.BENIGN_SOURCE == "all_files" and missing:
        sys.exit(
            f"\n[LỖI] BENIGN_SOURCE='all_files' cần đủ 8 file, đang thiếu "
            f"{len(missing)}:\n    "
            + "\n    ".join(missing)
            + f"\n\nHoặc tải lại bộ zip cho đủ, hoặc đặt BENIGN_SOURCE='same_file' "
            f"trong config.py\n(chỉ cần một file {C.WEB_ATTACK_FILE})."
        )

    if missing:
        print(
            f"\n[CHẤP NHẬN ĐƯỢC] Thiếu {len(missing)} file, nhưng "
            f"BENIGN_SOURCE='same_file' chỉ cần\n                 "
            f"{C.WEB_ATTACK_FILE} — đã có mặt."
        )
        print(
            "                 Các file thiếu chỉ chứa tấn công phi-web, "
            "vốn nằm ngoài phạm vi đề tài:"
        )
        for m in missing:
            print(f"                   - {m}")
    else:
        print("\n[ĐẠT] Đủ cả 8 file.")


def check_web_attack_file(strict: bool = True) -> None:
    """Đọc thật file tấn công web và đối chiếu số mẫu từng lớp."""
    path = C.RAW_DIR / C.WEB_ATTACK_FILE
    print("\n" + "=" * 78)
    print(f"ĐỌC THỬ FILE TẤN CÔNG WEB")
    print("=" * 78)
    print(f"  {path.name}")

    try:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
    except Exception as e:
        sys.exit(
            f"\n[LỖI] Không đọc được file: {e}\n"
            f"       File nhiều khả năng bị cắt cụt do giải nén lỗi. "
            f"Hãy tải và giải nén lại."
        )

    df.columns = [c.strip() for c in df.columns]
    print(f"  Kích thước: {len(df):,} dòng x {df.shape[1]} cột")

    if "Label" not in df.columns:
        sys.exit(
            f"\n[LỖI] Không tìm thấy cột 'Label'. Các cột hiện có: "
            f"{list(df.columns)[:10]}..."
        )

    print("\n  Nhãn thô (repr, để lộ byte hỏng nếu có):")
    for raw, n in df["Label"].value_counts().items():
        print(f"    {n:>8,}  {raw!r}")

    dist = distribution(df["Label"])
    print("\n  Sau khi chuẩn hóa qua labels.to_group():")
    for name, n in dist.most_common():
        print(f"    {n:>8,}  {name}")

    print("\n  Đối chiếu với số mẫu kỳ vọng của CIC-IDS2017:")
    all_ok = True
    for name, want in C.EXPECTED_WEB_COUNTS.items():
        got = dist.get(name, 0)
        ok = got == want
        all_ok &= ok
        print(
            f"    [{'ĐẠT ' if ok else 'LỆCH'}] {name:<16} "
            f"kỳ vọng {want:>6,}   thực tế {got:>6,}"
        )

    total_web = sum(dist.get(k, 0) for k in C.WEB_ATTACK_CLASSES)
    print(f"\n  Tổng mẫu tấn công web : {total_web:,}")
    print(f"  Mẫu BENIGN cùng file  : {dist.get('BENIGN', 0):,}")
    if dist.get("OTHER"):
        print(
            f"  Mẫu tấn công phi-web  : {dist['OTHER']:,}  (sẽ bị loại khi WEB_ONLY=True)"
        )

    if not all_ok:
        if strict:
            sys.exit(
                "\n[LỖI] Số mẫu tấn công web không khớp kỳ vọng.\n"
                "       File nhiều khả năng bị cắt cụt. Giải nén lại rồi chạy lại lệnh này.\n"
                "       (Nếu đang dùng dữ liệu GIẢ, chạy: "
                "python -m flow_based.data.verify_data --synthetic)"
            )
        print("\n[BỎ QUA] Cờ --synthetic đang bật nên không chặn sai lệch số mẫu.")

    if C.BENIGN_SOURCE == "same_file":
        ratio = dist.get("BENIGN", 0) / max(total_web, 1)
        print(
            f"\n  Tỷ số mất cân bằng dự kiến (BENIGN / tấn công web): {ratio:,.1f} : 1"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Kiểm tra dữ liệu trước tiền xử lý")
    parser.add_argument("--dataset", choices=list(C.DATASETS), default=None)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.dataset:
        C.select_dataset(args.dataset)
    if C.DATASET_ID == "cse_cicids2018":
        if args.synthetic:
            parser.error("Profile 2018 yêu cầu dữ liệu thật đã xác minh.")
        from flow_based.data.cse2018 import verify

        verify()
        return
    strict = not args.synthetic
    present, missing = check_files()
    check_required(missing)
    check_web_attack_file(strict=strict)
    print("\n" + "=" * 78)
    print(
        "KẾT LUẬN: Dữ liệu hợp lệ cho cấu hình hiện tại. "
        "Có thể chạy python -m flow_based.data.data_prep."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()

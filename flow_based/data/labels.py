"""
labels.py - Chuẩn hóa và ánh xạ nhãn CIC-IDS2017.

VÌ SAO CẦN MODULE RIÊNG
=======================
Đây là chỗ đã làm hỏng toàn bộ lần thực nghiệm trước.

Các file CSV của CIC-IDS2017 ghi nhãn tấn công web dưới dạng:

    b"Web Attack \\x96 Brute Force"

Byte 0x96 là dấu gạch ngang en-dash trong bảng mã cp1252, KHÔNG hợp lệ trong
UTF-8. Tùy cách đọc file mà nó biến thành nhiều dạng khác nhau:

    doc bang latin1      -> "Web Attack \\x96 Brute Force"
    doc bang cp1252      -> "Web Attack – Brute Force"      (en-dash U+2013)
    da hong tu truoc     -> "Web Attack \\ufffd Brute Force" (ky tu thay the)
    hong roi doc latin1  -> "Web Attack ï¿½ Brute Force"     (3 ky tu!)

Cách làm cũ so khớp CHUỖI TUYỆT ĐỐI với một bảng ánh xạ cứng, nên chỉ cần lệch
một byte là toàn bộ nhãn tấn công web rơi vào nhánh "không ánh xạ được". Hậu quả:
chỉ số web_macro_recall - đúng chỉ số trọng tâm của đề tài - bằng NaN suốt 5 lần
chạy mà không ai phát hiện, vì code chỉ in CẢNH BÁO rồi chạy tiếp.

CÁCH LÀM MỚI
============
1. normalize_label() quy MỌI ký tự không in được (kể cả chuỗi mojibake nhiều
   byte) về một dấu gạch ngang duy nhất -> không còn phụ thuộc byte cụ thể.
2. to_group() so khớp theo TỪ KHÓA ("brute" / "xss" / "sql"), không so khớp
   chuỗi tuyệt đối -> bền với mọi biến thể khoảng trắng và dấu câu.
3. assert_web_labels() DỪNG CỨNG nếu số mẫu không khớp kỳ vọng, thay vì in
   cảnh báo rồi chạy tiếp.
"""

from __future__ import annotations

import re
import sys
import unicodedata

# labels.py không import config.py (không cần) nên không tự động được vá lỗi
# encoding stdout ở đó khi chạy độc lập (`python -m flow_based.data.labels`) - vá riêng ở đây.
# Xem giải thích đầy đủ ở đầu config.py.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from collections import Counter
from typing import Iterable

# Chuỗi mojibake phổ biến khi byte 0x96 bị giải mã sai nhiều lần.
# Xử lý riêng vì chúng gồm các ký tự ASCII in được, regex bên dưới không bắt.
_MOJIBAKE = ("ï¿½", "â€“", "â€”", "Ã¢â‚¬â€œ")


def normalize_label(s) -> str:
    """Chuẩn hóa một nhãn thô về dạng ASCII sạch, ổn định.

    >>> normalize_label("Web Attack \\x96 Brute Force")
    'Web Attack - Brute Force'
    >>> normalize_label("Web Attack ï¿½ XSS")
    'Web Attack - XSS'
    >>> normalize_label("  BENIGN  ")
    'BENIGN'
    """
    s = str(s)
    # Thay chuỗi mojibake TRƯỚC khi chuẩn hóa Unicode. Nếu làm ngược lại, NFKC
    # tách "½" thành "1/2" và phá vỡ mẫu "ï¿½" trước khi ta kịp nhận ra nó.
    for bad in _MOJIBAKE:
        s = s.replace(bad, "-")
    # Chuẩn hóa Unicode để các dạng tổ hợp quy về một dạng duy nhất.
    s = unicodedata.normalize("NFKC", s)
    # MỌI ký tự ngoài vùng ASCII in được -> dấu gạch ngang.
    # Đây là điểm mấu chốt: không cần biết byte hỏng cụ thể là gì.
    s = re.sub(r"[^\x20-\x7E]+", "-", s)
    # Gộp mọi cụm gạch ngang + khoảng trắng quanh nó thành " - "
    s = re.sub(r"\s*-+\s*", " - ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_group(s) -> str:
    """Ánh xạ nhãn thô về nhóm lớp dùng trong đồ án.

    So khớp theo TỪ KHÓA nên bền với mọi biến thể mã hóa và khoảng trắng.
    Trả về "OTHER" cho các tấn công không thuộc tầng ứng dụng web
    (DoS, DDoS, PortScan, Bot, Infiltration, Heartbleed, Patator...).
    """
    low = normalize_label(s).lower()

    if low == "benign":
        return "BENIGN"

    # CSE-CIC-IDS2018: không gộp FTP/SSH brute force vào tấn công web.
    canonical = re.sub(r"[^a-z0-9]+", " ", low).strip()
    groups_2018 = {
        "brute force web": "Web-BruteForce",
        "brute force xss": "Web-XSS",
        "sql injection": "Web-SQLi",
    }
    if canonical in groups_2018:
        return groups_2018[canonical]

    if "web attack" in low:
        if "brute" in low:
            return "Web-BruteForce"
        if "xss" in low:
            return "Web-XSS"
        if "sql" in low:
            return "Web-SQLi"
        # Có nhãn "Web Attack" nhưng không rõ loại -> phải biết ngay, không nuốt.
        raise ValueError(
            f"Nhãn tấn công web không nhận dạng được: {s!r} -> {low!r}. "
            f"Cập nhật to_group() trong labels.py trước khi chạy tiếp."
        )

    return "OTHER"


def map_labels(raw_labels: Iterable) -> list[str]:
    """Ánh xạ cả một cột nhãn. Trả về list các tên nhóm."""
    return [to_group(x) for x in raw_labels]


def assert_web_labels(
    counts: dict[str, int], expected: dict[str, int], strict: bool = True
) -> None:
    """DỪNG CỨNG nếu số mẫu tấn công web không khớp kỳ vọng.

    Đây là chốt chặn thay cho cảnh báo im lặng của lần chạy trước.

    counts   : phân bố nhóm thực tế đếm được sau khi ánh xạ
    expected : config.EXPECTED_WEB_COUNTS
    strict   : True  -> sai lệch bất kỳ đều raise (dùng cho dữ liệu thật)
               False -> chỉ in cảnh báo (dùng cho dữ liệu giả khi smoke test)
    """
    problems = []
    for name, want in expected.items():
        got = counts.get(name, 0)
        if got != want:
            problems.append(f"  {name:<16} kỳ vọng {want:>6,}  thực tế {got:>6,}")

    if not problems:
        print(
            "[nhãn] Đạt: số mẫu tấn công web khớp hoàn toàn với kỳ vọng "
            f"({', '.join(f'{k}={v:,}' for k, v in expected.items())})"
        )
        return

    msg = (
        "Số mẫu tấn công web KHÔNG khớp kỳ vọng của CIC-IDS2017:\n"
        + "\n".join(problems)
        + "\n\nNguyên nhân thường gặp:\n"
        "  - Thiếu file Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv\n"
        "  - File CSV bị cắt cụt do giải nén lỗi (chạy python -m flow_based.data.verify_data để kiểm tra)\n"
        "  - Nhãn hỏng mã hóa mà normalize_label() chưa xử lý được\n"
        f"\nPhân bố nhóm thực tế đếm được: {dict(counts)}"
    )
    if strict:
        raise AssertionError(msg)
    print(f"[CẢNH BÁO] {msg}")


def distribution(labels: Iterable) -> Counter:
    """Đếm phân bố nhóm sau khi ánh xạ."""
    return Counter(map_labels(labels))


if __name__ == "__main__":
    # Kiểm tra nhanh mọi biến thể mã hóa đã biết đều quy về cùng một nhóm.
    variants = [
        "Web Attack \x96 Brute Force",  # byte 0x96 doc bang latin1
        "Web Attack \u2013 Brute Force",  # en-dash that
        "Web Attack \ufffd Brute Force",  # ky tu thay the
        "Web Attack ï¿½ Brute Force",  # mojibake 3 ky tu
        "Web Attack - Brute Force",  # da sach
        "Web Attack  -  Brute  Force",  # thua khoang trang
    ]
    print("Kiểm tra tính bền của normalize_label() + to_group():\n")
    ok = True
    for v in variants:
        g = to_group(v)
        flag = "OK " if g == "Web-BruteForce" else "SAI"
        if g != "Web-BruteForce":
            ok = False
        print(f"  [{flag}] {v!r:<45} -> {normalize_label(v)!r:<28} -> {g}")

    for v, want in [
        ("Web Attack \x96 XSS", "Web-XSS"),
        ("Web Attack \x96 Sql Injection", "Web-SQLi"),
        ("BENIGN", "BENIGN"),
        ("DoS Hulk", "OTHER"),
        ("PortScan", "OTHER"),
    ]:
        g = to_group(v)
        flag = "OK " if g == want else "SAI"
        if g != want:
            ok = False
        print(f"  [{flag}] {v!r:<45} -> {normalize_label(v)!r:<28} -> {g}")

    print(
        "\n"
        + ("TẤT CẢ ĐỀU ĐÚNG." if ok else "CÓ TRƯỜNG HỢP SAI - phải sửa trước khi chạy.")
    )

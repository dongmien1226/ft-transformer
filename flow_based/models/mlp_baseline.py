"""
mlp_baseline.py - Mạng nơ-ron truyền thẳng làm mô hình cơ sở (Chương 3, Bảng 3.6).

VÌ SAO TỰ VIẾT BẰNG PYTORCH THAY VÌ DÙNG sklearn.MLPClassifier
==============================================================
1. Bảng 3.6 yêu cầu khai báo *dropout*, *optimizer*, *batch size*, *số epoch*.
   sklearn.MLPClassifier không hỗ trợ dropout.

2. Quan trọng hơn - Chương 3 mục 3.2.3 nêu rõ vai trò của MLP:

       "cho phép tách biệt phần đóng góp đến từ việc sử dụng học sâu nói chung
        với phần đóng góp đến từ riêng cơ chế tự chú ý"

   Muốn kết luận đó có giá trị thì MLP và Transformer phải khác nhau ĐÚNG MỘT
   THỨ: kiến trúc. Mọi yếu tố còn lại - hàm mất mát có trọng số lớp, thuật toán
   tối ưu, scheduler, điều kiện dừng sớm, phép chia dữ liệu - phải giống hệt.
   Vì vậy MLP dùng CHUNG trainer.train_torch_model() với Transformer.

Điểm khác biệt kiến trúc cần nhấn mạnh khi phân tích kết quả: ở MLP, toàn bộ F
đặc trưng bị trộn lẫn ngay tại lớp tuyến tính đầu tiên theo một bộ trọng số cố
định, không phụ thuộc giá trị cụ thể của mẫu đang xét. Ở Transformer, trọng số
liên kết giữa các đặc trưng được tính lại cho từng mẫu.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    """Mạng truyền thẳng nhiều lớp: Linear -> Norm -> Activation -> Dropout."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        act = _ACTIVATIONS.get(activation)
        if act is None:
            raise ValueError(f"Hàm kích hoạt không hỗ trợ: {activation!r}")

        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), act(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))

        self.net = nn.Sequential(*layers)
        self.hidden_dims = hidden_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_mlp(n_features: int, n_classes: int, mlp_cfg: dict) -> MLP:
    return MLP(
        n_features=n_features,
        n_classes=n_classes,
        hidden_dims=mlp_cfg["hidden_dims"],
        dropout=mlp_cfg["dropout"],
        activation=mlp_cfg["activation"],
    )


if __name__ == "__main__":
    from flow_based import config as C

    m = build_mlp(68, 4, C.MLP)
    x = torch.randn(8, 68)
    y = m(x)
    n_par = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(m)
    print(f"\nĐầu vào: {tuple(x.shape)}  ->  Đầu ra: {tuple(y.shape)}")
    print(f"Tham số: {n_par:,}")

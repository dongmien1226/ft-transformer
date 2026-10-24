"""
ft_transformer.py - Kiến trúc Transformer đề xuất (Chương 2, mục 2.4).

Ý tưởng cốt lõi: COI MỖI ĐẶC TRƯNG CỦA LUỒNG MẠNG NHƯ MỘT TOKEN.

Khác với mạng nơ-ron truyền thẳng - nơi toàn bộ F đặc trưng bị trộn lẫn ngay ở
lớp đầu tiên theo một bộ trọng số cố định - ở đây mỗi đặc trưng được ánh xạ
độc lập thành một vectơ d chiều, rồi cơ chế tự chú ý cho phép từng đặc trưng
truy cập trực tiếp tới mọi đặc trưng khác với trọng số HỌC ĐƯỢC TỪ DỮ LIỆU.

    Token hóa:  e_j = x_j * w_j + b_j ,  w_j, b_j thuộc R^d ,  j = 1..F   (2.5)
    Chuỗi vào:  Z0 = [e_cls ; e_1 ; ... ; e_F]  thuộc R^((F+1) x d)
    Encoder:    N khối, tự chú ý đa đầu + truyền thẳng, Pre-LN
    Đầu ra:     y = W_out * GELU(LayerNorm(z_cls)) + b_out                (2.12)

KHÔNG DÙNG POSITIONAL ENCODING. Thứ tự các cột trong file CSV là quy ước tùy ý
của công cụ trích xuất đặc trưng, không mang ý nghĩa. Thêm positional encoding
sẽ gán cho mô hình một thông tin thứ tự giả tạo. Danh tính của từng đặc trưng
đã nằm sẵn trong bộ tham số riêng (w_j, b_j) của cột đó.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Biến vectơ F chiều thành chuỗi F token, mỗi token d chiều.

    e_j = x_j * w_j + b_j

    Mỗi cột có một cặp (w_j, b_j) riêng để biểu diễn danh tính đặc trưng.
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))

        # Khởi tạo theo kiểu Kaiming uniform quen thuộc của lớp tuyến tính.
        bound = 1.0 / (d_model**0.5)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)  ->  (B, F, d)
        e = x.unsqueeze(-1) * self.weight + self.bias
        return e

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, d_model={self.d_model}"


class FTTransformer(nn.Module):
    """Transformer cho dữ liệu bảng, theo đặc tả Chương 2 Bảng 2.7.

    Token hóa từng đặc trưng với tham số riêng cho mỗi cột, sau đó tổng hợp
    đầu ra encoder bằng token [CLS] để phân loại.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 128,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) phải chia hết cho n_heads ({n_heads})"
            )
        self.n_features = n_features
        self.tokenizer = FeatureTokenizer(n_features, d_model)

        # Token [CLS]: nội dung là tham số học được, đóng vai trò nơi tổng hợp
        # thông tin từ toàn bộ các token đặc trưng.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.n_tokens = n_features + 1  # +1 cho [CLS]

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,  # Pre-LN: ổn định, không cần warmup
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Đầu ra: y = W_out * GELU(LayerNorm(z_cls)) + b_out
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        z = self.tokenizer(x)  # (B, F, d)

        cls = self.cls_token.expand(z.size(0), -1, -1)  # (B, 1, d)
        z = torch.cat([cls, z], dim=1)  # (B, T+1, d)

        z = self.encoder(z)

        pooled = z[:, 0]  # chỉ lấy vị trí [CLS]
        return self.head(pooled)


def build_transformer(
    n_features: int, n_classes: int, model_cfg: dict
) -> FTTransformer:
    """Dựng mô hình từ dict cấu hình, để mọi nơi gọi đều dùng chung một nguồn."""
    return FTTransformer(
        n_features=n_features,
        n_classes=n_classes,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
        activation=model_cfg["activation"],
        norm_first=model_cfg["norm_first"],
    )


if __name__ == "__main__":
    from flow_based import config as C

    F_, Cn = 68, 4
    m = build_transformer(F_, Cn, C.MODEL)
    n_par = sum(p.numel() for p in m.parameters() if p.requires_grad)
    x = torch.randn(8, F_)
    y = m(x)

    print(m)
    print(f"\nĐầu vào  : {tuple(x.shape)}")
    print(f"Số token : {m.n_tokens} (= {F_} đặc trưng + 1 token [CLS])")
    print(f"Đầu ra   : {tuple(y.shape)}")
    print(f"Tham số  : {n_par:,}")

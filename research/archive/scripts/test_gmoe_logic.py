import torch
import torch.nn as nn
import hashlib


class GhostExpert(nn.Module):
    def __init__(self, in_features, out_features, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        u = torch.randn(out_features, 1, generator=gen, dtype=torch.bfloat16)
        v = torch.randn(1, in_features, generator=gen, dtype=torch.bfloat16)
        self.register_buffer("u", u / u.norm())
        self.register_buffer("v", v / v.norm())
        self.scale = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    def forward(self, x):
        # returns the delta only
        return (x @ self.v.T) @ self.u.T * self.scale


class GhostChainLayer(nn.Module):
    def __init__(self, base_layer, name):
        super().__init__()
        self.base_layer = base_layer
        out_features, in_features = base_layer.weight.shape

        # 3 Experts in series-ish
        base_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
        self.expert1 = GhostExpert(in_features, out_features, base_seed + 1)
        self.expert2 = GhostExpert(in_features, out_features, base_seed + 2)
        self.expert3 = GhostExpert(in_features, out_features, base_seed + 3)

        # 1 Observer
        self.observer = GhostExpert(in_features, out_features, base_seed + 4)

    def forward(self, x):
        # Serial capitalization: each expert can "see" the previous one's nudge
        d1 = self.expert1(x)
        d2 = self.expert2(x + d1)
        d3 = self.expert3(x + d2)

        # Observer benefits from the total success/failure of the 3
        total_nudge = d1 + d2 + d3
        correction = self.observer(x + total_nudge)

        return self.base_layer(x) + total_nudge + correction


# Test the setup
def test():
    base = nn.Linear(2560, 2560).to(dtype=torch.bfloat16)
    layer = GhostChainLayer(base, "layer.0.q_proj")
    x = torch.randn(1, 19, 2560, dtype=torch.bfloat16)
    y = layer(x)
    print(f"Output shape: {y.shape}")
    print(
        f"Trainable params in layer: {sum(p.numel() for p in layer.parameters() if p.requires_grad)}"
    )


if __name__ == "__main__":
    test()

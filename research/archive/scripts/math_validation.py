import torch
import torch.nn as nn
import torch.optim as optim


def simulate_learning(mode="parallel", steps=100):
    # Hidden target function: we want to nudge x to some target y'
    # For a simple test, let's say target = x + noise_target
    torch.manual_seed(42)
    dim = 2560
    x = torch.randn(1, dim)
    target_nudge = torch.randn(1, dim) * 0.1
    target = x + target_nudge

    # Components
    def get_expert():
        u = torch.randn(dim, 1)
        v = torch.randn(1, dim)
        u /= u.norm()
        v /= v.norm()
        s = nn.Parameter(torch.zeros(1))
        return u, v, s

    u1, v1, s1 = get_expert()
    u2, v2, s2 = get_expert()
    u3, v3, s3 = get_expert()
    uo, vo, so = get_expert()

    params = [s1, s2, s3, so]
    optimizer = optim.Adam(params, lr=0.1)

    history = []
    for _ in range(steps):
        optimizer.zero_grad()

        def run_expert(inp, u, v, s):
            return (inp @ v.T) @ u.T * s

        if mode == "parallel":
            d1 = run_expert(x, u1, v1, s1)
            d2 = run_expert(x, u2, v2, s2)
            d3 = run_expert(x, u3, v3, s3)
            do = run_expert(x, uo, vo, so)
            pred = x + d1 + d2 + d3 + do
        else:  # serial ghost chain
            d1 = run_expert(x, u1, v1, s1)
            d2 = run_expert(x + d1, u2, v2, s2)
            d3 = run_expert(x + d2, u3, v3, s3)
            total = d1 + d2 + d3
            do = run_expert(x + total, uo, vo, so)
            pred = x + total + do

        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return history


print("Testing Math: Parallel vs Serial Ghost Chain (100 steps)")
h_para = simulate_learning("parallel")
h_seri = simulate_learning("serial")

print(f"Parallel Final Loss: {h_para[-1]:.8f}")
print(f"Serial   Final Loss: {h_seri[-1]:.8f}")
print(f"Improvement: {(h_para[-1] - h_seri[-1]) / h_para[-1] * 100:.2f}%")

# Parameter Math
layers = 210
experts = 4
learned_params = layers * experts
storage_kb = (learned_params * 4) / 1024  # FP32
print(f"\nParameter Audit:")
print(f"Total learned parameters: {learned_params}")
print(f"Storage footprint: {storage_kb:.2f} KB")

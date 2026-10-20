"""Hand-checkable unit tests for the three core objectives (DPO, PPO clipping, GRPO advantages).

Run from the repository root:
    python -m scripts.test_objectives

Each test uses tiny tensors whose correct values can be worked out by hand from the equations
in the assignment manual, so a failure points directly at the objective rather than at training.
"""
from __future__ import annotations

import math

import torch

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss


def close(a, b, tol=1e-5):
    return abs(float(torch.as_tensor(a).detach()) - float(b)) < tol


# ----------------------------------------------------------------------------- DPO
def test_dpo_at_initialisation_is_log2():
    # When pi_theta == pi_ref the implicit reward margin is zero for every pair,
    # so the loss must be -log(sigmoid(0)) = log 2 regardless of the sequence log-probs.
    pc = torch.tensor([-12.0, -40.0, -3.0])
    pr = torch.tensor([-15.0, -20.0, -9.0])
    loss, stats = dpo_loss(pc, pr, pc.clone(), pr.clone(), beta=0.1)
    assert close(loss, math.log(2)), f"expected log2, got {loss}"
    assert close(stats["logit_mean"], 0.0)


def test_dpo_matches_manual_formula():
    # One pair: policy raised chosen by +2 nats and lowered rejected by -1 nat relative to the reference.
    # margin m = (pc - rc) - (pr - rr) = 2 - (-1) = 3;  loss = -log sigmoid(beta * 3).
    beta = 0.1
    rc, rr = torch.tensor([-10.0]), torch.tensor([-12.0])
    pc, pr = rc + 2.0, rr - 1.0
    loss, stats = dpo_loss(pc, pr, rc, rr, beta)
    expected = -math.log(1.0 / (1.0 + math.exp(-beta * 3.0)))
    assert close(loss, expected), f"expected {expected}, got {loss}"
    assert stats["preference_accuracy"].item() == 1.0


def test_dpo_gradient_direction():
    # Gradient descent on the loss must push chosen log-prob up and rejected log-prob down.
    pc = torch.tensor([-10.0], requires_grad=True)
    pr = torch.tensor([-10.0], requires_grad=True)
    loss, _ = dpo_loss(pc, pr, torch.tensor([-10.0]), torch.tensor([-10.0]), beta=0.1)
    loss.backward()
    assert pc.grad.item() < 0 and pr.grad.item() > 0


# ----------------------------------------------------------------------------- PPO
def test_ppo_clip_positive_advantage():
    # A = +1, eps = 0.2. rho = 1.5 is outside [0.8, 1.2]; the pessimistic objective is
    # min(1.5*1, 1.2*1) = 1.2, so the loss is -1.2 and its gradient w.r.t. rho is zero (clipped).
    old = torch.zeros(1, 1)
    new = torch.log(torch.tensor([[1.5]])).requires_grad_(True)
    adv = torch.ones(1, 1)
    mask = torch.ones(1, 1)
    loss, ratio, clip_frac = ppo_policy_loss(new, old, adv, mask, eps=0.2)
    assert close(loss, -1.2), f"expected -1.2, got {loss}"
    loss.backward()
    assert close(new.grad.item(), 0.0), "clipped token must not receive gradient"
    assert close(clip_frac, 1.0)


def test_ppo_clip_negative_advantage():
    # A = -1, rho = 0.5 (below 1-eps): min(0.5*-1, 0.8*-1) = min(-0.5, -0.8) = -0.8 -> loss 0.8, zero grad.
    # A = -1, rho = 1.5 (above 1+eps): min(-1.5, -1.2) = -1.5 -> loss 1.5, gradient still flows
    # (PPO never clips a move that makes a bad action *more* likely).
    old = torch.zeros(1, 2)
    new = torch.log(torch.tensor([[0.5, 1.5]])).requires_grad_(True)
    adv = -torch.ones(1, 2)
    mask = torch.ones(1, 2)
    loss, _, _ = ppo_policy_loss(new, old, adv, mask, eps=0.2)
    assert close(loss, (0.8 + 1.5) / 2), f"expected 1.15, got {loss}"
    loss.backward()
    assert close(new.grad[0, 0].item(), 0.0)
    assert abs(new.grad[0, 1].item()) > 0


def test_ppo_inside_trust_region_is_unclipped():
    old = torch.zeros(1, 1)
    new = torch.log(torch.tensor([[1.1]]))
    loss, _, clip_frac = ppo_policy_loss(new, old, torch.full((1, 1), 2.0), torch.ones(1, 1), eps=0.2)
    assert close(loss, -2.2) and close(clip_frac, 0.0)


def test_ppo_mask_ignores_padding():
    old = torch.zeros(1, 2)
    new = torch.log(torch.tensor([[1.0, 5.0]]))
    loss, _, clip_frac = ppo_policy_loss(new, old, torch.ones(1, 2), torch.tensor([[1.0, 0.0]]), eps=0.2)
    assert close(loss, -1.0) and close(clip_frac, 0.0)


def test_gae_matches_brute_force():
    # Right-padded batch with different response lengths, gamma/lambda < 1. Reference: the definition
    # A_t = sum_k (gamma*lambda)^k delta_{t+k}, delta_t = r_t + gamma*V_{t+1} - V_t, with V = 0 after the last token.
    torch.manual_seed(0)
    lens, gamma, lam = [7, 4, 1], 0.97, 0.9
    mask = torch.zeros(3, 7)
    for b, l in enumerate(lens):
        mask[b, :l] = 1
    r, v = torch.randn(3, 7) * mask, torch.randn(3, 7)
    adv, ret = compute_gae(r, v, mask, gamma, lam)
    ref = torch.zeros(3, 7)
    for b, l in enumerate(lens):
        for t in range(l):
            ref[b, t] = sum((gamma * lam) ** (k - t) * (r[b, k] + gamma * (v[b, k + 1] if k + 1 < l else 0.0) - v[b, k])
                            for k in range(t, l))
    assert torch.allclose(adv * mask, ref, atol=1e-5)
    assert torch.allclose(ret, adv + v, atol=1e-6)


def test_shaped_rewards_kl_and_terminal():
    # r_t = -beta_KL (log pi - log pi_ref) on every valid token, + task reward on the last valid token only.
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    pl, rl = torch.tensor([[-1.0, -2.0, -0.5], [-1.0, 0.0, 0.0]]), torch.tensor([[-1.5, -2.0, -1.0], [-0.5, 0.0, 0.0]])
    out = shaped_rewards(torch.tensor([2.0, -1.0]), pl, rl, mask, beta_kl=0.1)
    expected = torch.tensor([[-0.05, 0.0, -0.05 + 2.0], [0.05 - 1.0, 0.0, 0.0]])
    assert torch.allclose(out, expected, atol=1e-6), out


def test_gae_terminal_reward():
    # gamma = 1, lambda = 1, V = 0: every position's advantage equals the terminal reward.
    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    values = torch.zeros(1, 4)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.tensor([[1.0, 1.0, 1.0, 0.0]]))


# ----------------------------------------------------------------------------- GRPO
def test_grpo_advantages_are_per_group():
    # Group 0: rewards [1, 0] -> mean 0.5, std 0.5 -> advantages [+1, -1].
    # Group 1: rewards [5, 5] -> std 0 -> advantages [0, 0] (an uninformative group).
    rewards = torch.tensor([1.0, 0.0, 5.0, 5.0])
    groups = torch.tensor([0, 0, 1, 1])
    adv = group_relative_advantages(rewards, groups)
    assert torch.allclose(adv, torch.tensor([1.0, -1.0, 0.0, 0.0]), atol=1e-4), adv


def test_grpo_advantages_invariant_to_other_groups():
    # Adding a constant to another prompt's rewards must not change this prompt's advantages.
    a = group_relative_advantages(torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 1.0]), torch.tensor([0, 0, 0, 1, 1, 1]))
    b = group_relative_advantages(torch.tensor([1.0, 2.0, 3.0, 100.0, 100.0, 101.0]), torch.tensor([0, 0, 0, 1, 1, 1]))
    assert torch.allclose(a, b, atol=1e-5)
    for g in range(2):
        assert close(a[3 * g: 3 * g + 3].mean(), 0.0)


def test_grpo_loss_normalisations():
    # Two completions, ratio = 1 everywhere, beta = 0 so loss = -mean_k(sum_t A_k * mask / norm).
    # Completion 0: 2 valid tokens, A = +1. Completion 1: 4 valid tokens, A = -1.
    new = torch.zeros(2, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    adv = torch.tensor([1.0, -1.0])
    # canonical GRPO: per-sequence 1/T_k -> (+1) and (-1) -> mean 0 -> loss 0
    loss, _ = grpo_policy_loss(new, new, adv, mask, new, eps=0.2, beta=0.0, loss_type="grpo")
    assert close(loss, 0.0)
    # Dr. GRPO: constant 1/L with L=4 -> (+2/4) and (-4/4) -> mean -0.25 -> loss +0.25
    loss, _ = grpo_policy_loss(new, new, adv, mask, new, eps=0.2, beta=0.0, loss_type="dr_grpo",
                               max_completion_length=4)
    assert close(loss, 0.25)


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

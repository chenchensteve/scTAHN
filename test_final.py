"""Pre-run checks for the final scTAHN code. Usage: python test_final.py"""
import numpy as np
import torch

from ppo import PPOAgent
from model import MoCo, GATEncoder


def test_ppo_shapes_and_credit():
    torch.manual_seed(0)
    agent = PPOAgent(state_dim=5, action_dim=1)
    agent.k_epochs = 1
    for i in range(7):
        agent.select_action(np.random.rand(5).astype(np.float32))
        agent.store_transition(float(i), None, False)

    # 记录各量形状
    states = torch.cat([x[0] for x in agent.memory])
    actions = torch.cat([x[1] for x in agent.memory])
    mu, std, v = agent.policy(states)
    lp = torch.distributions.Normal(mu, std).log_prob(actions).sum(dim=-1)
    assert lp.shape == (7,), lp.shape

    # 每条经验的策略梯度只取决于它自己的优势值：只给第 0 条非零优势，其余条目的 log_prob 梯度应为 0
    lp = lp.detach().requires_grad_(True)
    adv = torch.zeros(7, device=lp.device); adv[0] = 1.0
    ratios = torch.exp(lp - lp.detach())
    surr = torch.min(ratios * adv, torch.clamp(ratios, 0.8, 1.2) * adv)
    (-surr).mean().backward()
    assert lp.grad[0] != 0 and torch.all(lp.grad[1:] == 0), lp.grad

    agent.update()  # 内部形状断言
    assert agent.memory == []
    print("PPO: OK")


def test_queue_wraparound():
    m = MoCo(GATEncoder, num_genes=8, dim=16, out_dim=4, r=10)
    m.queue_ptr[0] = 8
    keys = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    m._dequeue_and_enqueue(keys, torch.tensor([100, 101, 102]))
    assert m.queue_idx[[8, 9, 0]].tolist() == [100, 101, 102]
    assert int(m.queue_ptr) == 1
    assert torch.allclose(m.queue[:, [8, 9, 0]], keys.T)

    # 连续写入后，队列中的索引恰好是最近 r 个入队样本
    m = MoCo(GATEncoder, num_genes=8, dim=16, out_dim=4, r=10)
    seen = []
    for step in range(9):
        idx = torch.arange(step * 3, step * 3 + 3)
        m._dequeue_and_enqueue(torch.nn.functional.normalize(torch.randn(3, 4), dim=1), idx)
        seen += idx.tolist()
    assert sorted(m.queue_idx.tolist()) == sorted(seen[-10:])
    print("Queue: OK")


if __name__ == "__main__":
    test_ppo_shapes_and_credit()
    test_queue_wraparound()

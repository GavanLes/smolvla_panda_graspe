import os
import math
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps

from mujoco_env.my_env import SimpleEnv


# =========================
# 0) 超参（先用这套跑通）
# =========================
SEED = 0
MAX_EP_STEPS = 250
HZ = 20

# residual 限幅（强烈建议小一点，避免抖动）
XI = 0.03          # residual 的最大幅度（单位跟你的 joint_angle 一致）
EXPL_NOISE = 0.01  # 探索噪声 std（同单位）

OFFLINE_SUCCESS_EP = 20    # 收集多少条成功 episode 进 offline buffer
WARMUP_STEPS = 2000        # 前多少步只用 base（online buffer 预热）
TOTAL_ENV_STEPS = 200_000  # 总交互步数

BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2               # 固定 entropy 系数（先别自动调，简单）
UPDATES_PER_STEP = 1


OFFLINE_PATH = "offline_success.npz"
SAVE_PATH = "residual_actor.pt"

TEXT_DIM = 64

device_vla = "cuda" if torch.cuda.is_available() else "cpu"
device_rl = "cpu"  # residual MLP 放 CPU 就行（你 env 很慢，主要耗在 VLA 推理）


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def text_to_feat(text: str, dim=64):
    v = np.zeros(dim, dtype=np.float32)
    if text is None:
        return v
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 1.0
    n = np.linalg.norm(v) + 1e-6
    return v / n


# =========================
# 1) Replay Buffer（两份）
# =========================
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size=250_000):
        self.max_size = max_size
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]).to(device_rl),
            torch.from_numpy(self.act[idx]).to(device_rl),
            torch.from_numpy(self.rew[idx]).to(device_rl),
            torch.from_numpy(self.next_obs[idx]).to(device_rl),
            torch.from_numpy(self.done[idx]).to(device_rl),
        )


def sample_mixed(offline_buf, online_buf, batch_size):
    half = batch_size // 2
    o1 = offline_buf.sample(half)
    o2 = online_buf.sample(batch_size - half)
    return tuple(torch.cat([a, b], dim=0) for a, b in zip(o1, o2))


# =========================
# 2) SAC 网络（MLP）
# =========================
LOG_STD_MIN = -5
LOG_STD_MAX = 2


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre_tanh = mu + eps * std
        a = torch.tanh(pre_tanh)  # in [-1,1]
        # tanh correction
        logp = -0.5 * (((pre_tanh - mu) / (std + 1e-6)) ** 2 + 2 * log_std + math.log(2 * math.pi))
        logp = logp.sum(dim=-1, keepdim=True)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return a, logp, torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


@torch.no_grad()
def soft_update(tgt: nn.Module, src: nn.Module, tau: float):
    for p_t, p in zip(tgt.parameters(), src.parameters()):
        p_t.data.mul_(1 - tau).add_(tau * p.data)


# =========================
# 3) 你的 env + VLA：算 a_base & 组 residual obs
# =========================
IMG_TF = transforms.Compose([transforms.ToTensor()])


@torch.no_grad()
def get_base_action(env: SimpleEnv, vla: SmolVLAPolicy):
    state8 = env.get_joint_state()[:8].astype(np.float32)
    img, wrist = env.grab_image()
    img_t = IMG_TF(Image.fromarray(img).resize((256, 256))).unsqueeze(0).to(device_vla)
    wrist_t = IMG_TF(Image.fromarray(wrist).resize((256, 256))).unsqueeze(0).to(device_vla)
    state_t = torch.from_numpy(state8).unsqueeze(0).to(device_vla)

    instr = getattr(env, "instruction", "")
    data = {
        "observation.state": state_t,
        "observation.image": img_t,
        "observation.wrist_image": wrist_t,
        "task": [instr],
    }
    a = vla.select_action(data)[0, :8].detach().cpu().numpy().astype(np.float32)  # 8维：含夹爪
    return state8, a, instr


def make_residual_obs(state8, a_base8, instr):
    # residual policy 的观测：state(8) + a_base(8) + instr_feat(64)
    return np.concatenate([state8, a_base8, text_to_feat(instr, TEXT_DIM)], axis=0).astype(np.float32)


def env_step_one_control_tick(env: SimpleEnv, a_exec8, do_render=False):
    # 执行一次控制
    _ = env.step(a_exec8)
    if do_render:
        env.render()

    # 推进仿真到下一个控制时刻（与你之前 loop_every 对齐）
    for _ in range(200):
        env.step_env()
        if env.env.loop_every(HZ=HZ):
            break


# =========================
# 4) 收集 offline 成功轨迹（base only）
# =========================
def collect_offline_success(env, vla, obs_dim, act_dim):
    buf = ReplayBuffer(obs_dim, act_dim, max_size=250_000)

    vla.reset()
    succ = 0
    tries = 0

    while succ < OFFLINE_SUCCESS_EP:
        tries += 1
        env.reset(seed=np.random.randint(0, 10_000))
        vla.reset()

        # 暂存一个 episode，成功才 commit
        ep = []
        for t in range(MAX_EP_STEPS):
            s, a_base, instr = get_base_action(env, vla)
            obs = make_residual_obs(s, a_base, instr)

            # base only => residual = 0
            delta = np.zeros((7,), dtype=np.float32)
            a_exec = a_base.copy()
            a_exec[:7] = np.clip(a_exec[:7] + delta, -np.inf, np.inf)  # 这里不额外限幅

            env_step_one_control_tick(env, a_exec, do_render=True)

            # next
            s2, a_base2, instr2 = get_base_action(env, vla)
            next_obs = make_residual_obs(s2, a_base2, instr2)

            success = bool(env.check_success())
            done = success or (t + 1 >= MAX_EP_STEPS)

            # 建议对齐论文：稀疏奖励
            rew = 1.0 if success else 0.0

            ep.append((obs, delta, rew, next_obs, float(done)))

            if done:
                if success:
                    for tr in ep:
                        buf.add(*tr)
                    succ += 1
                    print(f"[offline] success {succ}/{OFFLINE_SUCCESS_EP} (tries={tries})")
                break

    np.savez(
        OFFLINE_PATH,
        obs=buf.obs[:buf.size],
        act=buf.act[:buf.size],
        rew=buf.rew[:buf.size],
        next_obs=buf.next_obs[:buf.size],
        done=buf.done[:buf.size],
    )
    print(f"[offline] saved to {OFFLINE_PATH}, transitions={buf.size}")


def load_offline(obs_dim, act_dim):
    d = np.load(OFFLINE_PATH)
    buf = ReplayBuffer(obs_dim, act_dim, max_size=250_000)
    n = d["obs"].shape[0]
    for i in range(n):
        buf.add(d["obs"][i], d["act"][i], d["rew"][i], d["next_obs"][i], d["done"][i])
    return buf


# =========================
# 5) 主训练：Residual SAC + 双buffer 50/50
# =========================
def train_residual(env, vla, offline_buf, obs_dim, act_dim):
    online_buf = ReplayBuffer(obs_dim, act_dim, max_size=250_000)

    actor = Actor(obs_dim, act_dim).to(device_rl)
    critic = Critic(obs_dim, act_dim).to(device_rl)
    critic_tgt = Critic(obs_dim, act_dim).to(device_rl)
    critic_tgt.load_state_dict(critic.state_dict())

    opt_a = optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = optim.Adam(critic.parameters(), lr=3e-4)

    # ===================== ✅ 这里加：定期保存设置（放在 for 循环前） =====================
    SAVE_EVERY = 10_000
    SAVE_DIR = "./rl_ckpt"
    os.makedirs(SAVE_DIR, exist_ok=True)

    def save_actor(step):
        path_step = os.path.join(SAVE_DIR, f"residual_actor_step{step}.pt")
        path_latest = os.path.join(SAVE_DIR, "residual_actor_latest.pt")
        torch.save(actor.state_dict(), path_step)
        torch.save(actor.state_dict(), path_latest)
        print(f"[save] actor -> {path_step}")
    # ===============================================================================

    vla.reset()
    env.reset(seed=SEED)
    vla.reset()

    # 初始化 obs
    s, a_base, instr = get_base_action(env, vla)
    obs = make_residual_obs(s, a_base, instr)

    ep_steps = 0
    ep_succ = 0

    # ===================== ✅ 可选：防止中断丢进度（强烈建议） =====================
    try:
        for step in range(1, TOTAL_ENV_STEPS + 1):
            ep_steps += 1

            # 选 residual 动作
            if step <= WARMUP_STEPS:
                delta = np.zeros((7,), dtype=np.float32)
            else:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device_rl)
                    a01, _, a_det = actor.sample(obs_t)  # a01 in [-1,1]
                    delta = a01[0].cpu().numpy().astype(np.float32) * XI
                    delta += np.random.randn(7).astype(np.float32) * EXPL_NOISE
                    delta = np.clip(delta, -XI, XI)

            # 执行动作：a_exec = a_base + delta（不训练夹爪）
            a_exec = a_base.copy()
            a_exec[:7] = a_exec[:7] + delta

            env_step_one_control_tick(env, a_exec, do_render=True)

            # next
            s2, a_base2, instr2 = get_base_action(env, vla)
            next_obs = make_residual_obs(s2, a_base2, instr2)

            success = bool(env.check_success())
            done = success or (ep_steps >= MAX_EP_STEPS)
            rew = 1.0 if success else 0.0

            online_buf.add(obs, delta, rew, next_obs, float(done))

            obs = next_obs
            a_base = a_base2

            if done:
                if success:
                    ep_succ += 1
                env.reset(seed=np.random.randint(0, 10_000))
                vla.reset()
                s, a_base, instr = get_base_action(env, vla)
                obs = make_residual_obs(s, a_base, instr)
                ep_steps = 0

            # ====== 更新 SAC ======
            if step > WARMUP_STEPS and online_buf.size >= BATCH_SIZE and offline_buf.size >= BATCH_SIZE:
                for _ in range(UPDATES_PER_STEP):
                    o, a, r, o2, d = sample_mixed(offline_buf, online_buf, BATCH_SIZE)

                    with torch.no_grad():
                        a2, logp2, _ = actor.sample(o2)
                        a2 = a2 * XI
                        q1_t, q2_t = critic_tgt(o2, a2)
                        q_t = torch.min(q1_t, q2_t) - ALPHA * logp2
                        y = r + (1.0 - d) * GAMMA * q_t

                    q1, q2 = critic(o, a)
                    loss_c = ((q1 - y).pow(2).mean() + (q2 - y).pow(2).mean())

                    opt_c.zero_grad()
                    loss_c.backward()
                    opt_c.step()

                    a_pi, logp, _ = actor.sample(o)
                    a_pi = a_pi * XI
                    q1_pi, q2_pi = critic(o, a_pi)
                    q_pi = torch.min(q1_pi, q2_pi)
                    loss_a = (ALPHA * logp - q_pi).mean()

                    opt_a.zero_grad()
                    loss_a.backward()
                    opt_a.step()

                    soft_update(critic_tgt, critic, TAU)

            if step % 1000 == 0:
                print(f"[step {step}] online={online_buf.size} offline={offline_buf.size} succ_ep={ep_succ}")

            # ===================== ✅ 这里加：每 1w step 保存一次 =====================
            if step % SAVE_EVERY == 0:
                save_actor(step)
            # =======================================================================
    finally:
        # 任何中断（Ctrl+C）也会保存一份 latest
        save_actor(step if 'step' in locals() else 0)
    # ===============================================================================

    torch.save(actor.state_dict(), SAVE_PATH)
    print(f"[done] saved residual actor to {SAVE_PATH}")


# =========================
# 6) main
# =========================
if __name__ == "__main__":
    set_seed(SEED)

    # ---- load dataset stats & VLA ----
    try:
        dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root="./panda_demo_data")
    except:
        dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root="./omy_pnp_languagpanda_e")

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    cfg = SmolVLAConfig(input_features=input_features, output_features=output_features, chunk_size=5, n_action_steps=1)
    _ = resolve_delta_timestamps(cfg, dataset_metadata)

    vla = SmolVLAPolicy.from_pretrained(
        "./ckpt/smolvla_ramdom/035000/pretrained_model", #./ckpt/pretrained_model 第一次尝试的模型 80条随机数据
        dataset_stats=dataset_metadata.stats
    ).to(device_vla)
    vla.eval()

    # ---- env ----
    env = SimpleEnv("./asset/my_scene.xml", action_type="joint_angle")

    # residual obs/action dim
    obs_dim = 8 + 8 + TEXT_DIM
    act_dim = 7

    # ---- offline ----
    collect_offline_success(env, vla, obs_dim, act_dim)
    offline_buf = load_offline(obs_dim, act_dim)

    # ---- train ----
    train_residual(env, vla, offline_buf, obs_dim, act_dim)

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
from torchvision import transforms
import hashlib

from stable_baselines3 import SAC

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps

from mujoco_env.my_env import SimpleEnv


class PnPResidualEnv(gym.Env):
    def __init__(self, pnp_env, vla_policy, device="cuda",
                 max_steps=200, residual_scale=0.01, hz=20, render=False,
                 img_feat_size=32, text_feat_dim=64):
        super().__init__()
        self.env = pnp_env
        self.policy = vla_policy
        self.device = device

        self.max_steps = max_steps
        self.residual_scale = residual_scale
        self.hz = hz
        self.do_render = render

        self.img_feat_size = img_feat_size
        self.text_feat_dim = text_feat_dim
        self.img_tf = transforms.Compose([transforms.ToTensor()])

        # 缓存：保证 PPO 的 obs 对应“当前时刻”
        self.cur_state = np.zeros(8, dtype=np.float32)
        self.cur_a_vla = np.zeros(8, dtype=np.float32)
        self.cur_img_feat = np.zeros(img_feat_size * img_feat_size, dtype=np.float32)
        self.cur_wrist_feat = np.zeros(img_feat_size * img_feat_size, dtype=np.float32)
        self.cur_text_feat = np.zeros(text_feat_dim, dtype=np.float32)

        # obs = state(8) + vla_action(8) + img(32x32) + wrist(32x32) + text(64)
        obs_dim = 8 + 8 + (img_feat_size * img_feat_size) * 2 + text_feat_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # residual action
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        self.step_count = 0
        self.policy.to(self.device)
        self.policy.eval()

    def _img_to_feat(self, img_arr: np.ndarray) -> np.ndarray:
        # 256x256x3 -> 32x32 灰度 -> 1024 float
        pil = Image.fromarray(img_arr).convert("L").resize((self.img_feat_size, self.img_feat_size))
        feat = np.asarray(pil, dtype=np.float32).reshape(-1) / 255.0
        return feat

    def _text_to_feat(self, text: str) -> np.ndarray:
        # 最简：hashing trick（把词哈希到固定维度向量）
        v = np.zeros(self.text_feat_dim, dtype=np.float32)
        if text is None:
            return v
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self.text_feat_dim] += 1.0
        n = np.linalg.norm(v) + 1e-6
        return v / n

    @torch.no_grad()
    def _update_cache(self):
        # 当前 state
        self.cur_state = self.env.get_joint_state()[:8].astype(np.float32)

        # 当前图像
        image, wrist_image = self.env.grab_image()

        # 给 PPO 的视觉特征（轻量）
        self.cur_img_feat = self._img_to_feat(image)
        self.cur_wrist_feat = self._img_to_feat(wrist_image)

        # 给 PPO 的指令特征（轻量）
        instr = getattr(self.env, "instruction", "")
        self.cur_text_feat = self._text_to_feat(instr)

        # 给 VLA 的输入（原分辨率流程）
        img_t = self.img_tf(Image.fromarray(image).resize((256, 256))).unsqueeze(0).to(self.device)
        wrist_t = self.img_tf(Image.fromarray(wrist_image).resize((256, 256))).unsqueeze(0).to(self.device)
        state_t = torch.from_numpy(self.cur_state).unsqueeze(0).to(self.device)

        data = {
            "observation.state": state_t,
            "observation.image": img_t,
            "observation.wrist_image": wrist_t,
            "task": [instr],
        }
        a = self.policy.select_action(data)
        self.cur_a_vla = a[0, :8].detach().cpu().numpy().astype(np.float32)

    def _get_obs(self):
        return np.concatenate(
            [self.cur_state, self.cur_a_vla, self.cur_img_feat, self.cur_wrist_feat, self.cur_text_feat],
            axis=0
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.policy.reset()
        self.env.reset(seed=0 if seed is None else seed)

        # 初始化缓存（保证 PPO 第一次看到就有图像/指令）
        self._update_cache()
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # residual
        a = self.cur_a_vla.copy()
        #print("VLA action:", a )  # 打印原始 VLA 动作，观察范围/分布
        residual7 = np.asarray(action, dtype=np.float32) * self.residual_scale
        #print("Residual:", residual7 )  # 打印残差，观察范围/分布
        a[:7] += residual7          # ✅ 只修正7个关节
        #print("After residual:", a )  # 打印修正后动作，观察范围/分布
        

        # 限幅（避免抖动/打满）
        try:
            ctrl_range = self.env.env.model.actuator_ctrlrange[:8]  # (8,2)
            a = np.clip(a, ctrl_range[:, 0], ctrl_range[:, 1])
        except Exception:
            pass

        _ = self.env.step(a)
        if self.do_render:
            self.env.render()

        # 推进到下一个控制时刻（与你原来 loop_every 的节奏一致）
        for _ in range(200):
            self.env.step_env()
            if self.env.env.loop_every(HZ=self.hz):
                break

        # 更新下一时刻的缓存（下一步 PPO 就能看到新的图像/指令）
        self._update_cache()
        obs = self._get_obs()

        success = bool(self.env.check_success())
        terminated = success
        truncated = self.step_count >= self.max_steps
        reward = 1.0 if success else -0.01

        info = {"success": success}
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root="./panda_demo_data")
    except:
        dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root="./omy_pnp_languagpanda_e")

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    cfg = SmolVLAConfig(input_features=input_features, output_features=output_features,
                        chunk_size=5, n_action_steps=1)
    _ = resolve_delta_timestamps(cfg, dataset_metadata)

    policy = SmolVLAPolicy.from_pretrained(
        "./ckpt/smolvla/20000/pretrained_model",
        dataset_stats=dataset_metadata.stats
    )
    policy.to(device)
    policy.eval()

    xml_path = "./asset/my_scene.xml"
    PnPEnv = SimpleEnv(xml_path, action_type="joint_angle")

    rl_env = PnPResidualEnv(
        pnp_env=PnPEnv,
        vla_policy=policy,
        device=device,
        max_steps=250,
        residual_scale=0.02,
        hz=20,
        render=True,          # 训练先关掉窗口
        img_feat_size=32,
        text_feat_dim=64
    )

    model = SAC(
        "MlpPolicy",
        rl_env,
        verbose=1,
        device="cpu",
        n_steps=1000,
        batch_size=64
    )
    model.learn(total_timesteps=200_000, progress_bar=True, log_interval=1)
    model.save("ppo_residual_on_vla")

import os, json
import numpy as np
import pandas as pd
from PIL import Image

# ======= 你需要改的路径 =======
ROOT = "./demo_data_cup_ramdom"  # 你的 dataset 根目录
EP_PATH = os.path.join(ROOT, "data/chunk-000/episode_000000.parquet")
TASKS_PATH = os.path.join(ROOT, "meta/tasks.jsonl")
OUT_DIR = "./vlm_export"         # 输出目录
# ============================

os.makedirs(OUT_DIR, exist_ok=True)
img_dir = os.path.join(OUT_DIR, "images")
os.makedirs(img_dir, exist_ok=True)

# 1) 读取 task_index -> task 文本映射
task_map = {}
with open(TASKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        task_map[int(obj["task_index"])] = obj["task"]

# 2) 读取 episode parquet
df = pd.read_parquet(EP_PATH)
print("len:", len(df), "cols:", list(df.columns))

# 3) 一个“先跑通”的子任务阶段机（你后面再调阈值）
SUBTASK = {
    0: "approach the mug",
    1: "grasp the mug",
    2: "lift the mug",
    3: "move to the plate",
    4: "place the mug on the plate",
    5: "retreat"
}

def infer_stage(state6, action7, obj_init9, prev_stage):
    ee = np.asarray(state6[:3], dtype=np.float32)
    mug = np.asarray(obj_init9[:3], dtype=np.float32)
    plate = np.asarray(obj_init9[6:9], dtype=np.float32)
    g = float(action7[-1])

    d_e2m = np.linalg.norm(ee - mug)
    d_e2p = np.linalg.norm(ee - plate)

    # 下面阈值是“先跑通”的默认值，你后面会根据实际数值调
    if prev_stage == 0 and d_e2m < 0.06:
        return 1
    if prev_stage == 1 and g < 0.2:
        return 2
    if prev_stage == 2 and ee[2] > mug[2] + 0.06:
        return 3
    if prev_stage == 3 and d_e2p < 0.08:
        return 4
    if prev_stage == 4 and g > 0.8:
        return 5
    return prev_stage

# 4) 导出：只在“阶段变化点”取样（减少重复样本）
samples = []
prev_stage = 0

# episode 的 task_index（一般整条一致）
task_idx = int(df["task_index"].iloc[0])
task_text = task_map.get(task_idx, f"task_index={task_idx}")

for i, row in df.iterrows():
    stage = infer_stage(row["observation.state"], row["action"], row["obj_init"], prev_stage)
    if stage != prev_stage:
        # 保存两张图
        rgb = row["observation.image"]
        wrist = row["observation.wrist_image"]

        rgb_path = os.path.join(img_dir, f"ep0_f{i:04d}_rgb.png")
        wrist_path = os.path.join(img_dir, f"ep0_f{i:04d}_wrist.png")

        Image.fromarray(rgb).save(rgb_path)
        Image.fromarray(wrist).save(wrist_path)

        samples.append({
            "instruction": (
                "You are a robot planner. Given the task and current observation, "
                "output the next SUBTASK.\n"
                f"Task: {task_text}\n"
            ),
            "input": "<image><image>",
            "output": f"SUBTASK: {SUBTASK[stage]}",
            "images": [rgb_path, wrist_path]
        })

    prev_stage = stage

out_json = os.path.join(OUT_DIR, "vlm_train.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print("Exported samples:", len(samples))
print("Saved to:", out_json)

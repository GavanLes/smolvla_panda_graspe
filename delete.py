import os
import json
import shutil

DATASET_DIR = "/home/gavan/lerobot-mujoco-tutorial/demo_data"    # ← 这里改成你的数据集目录
EP_MIN_LENGTH = 5                        # 长度 <=5 的 episode 会被删除

episodes_dir = os.path.join(DATASET_DIR, "episodes")
meta_dir = os.path.join(DATASET_DIR, "meta")

# 1) 读取 episodes_stats.jsonl
stats_path = os.path.join(meta_dir, "episodes_stats.jsonl")
clean_stats_lines = []
delete_ep_ids = []

with open(stats_path, "r") as f:
    for line in f.readlines():
        data = json.loads(line)
        ep_id = data["episode_id"]
        length = data["length"]

        if length <= EP_MIN_LENGTH:
            print(f"[DELETE] episode {ep_id} length={length}")
            delete_ep_ids.append(ep_id)
        else:
            clean_stats_lines.append(line)

# 2) 删除对应 parquet 文件
for ep_id in delete_ep_ids:
    ep_file = os.path.join(episodes_dir, f"episode_{ep_id:06d}.parquet")
    if os.path.exists(ep_file):
        os.remove(ep_file)
        print(f"Removed file: {ep_file}")

# 3) 清理 meta/episodes.jsonl（保持 meta 一致）
episodes_meta_path = os.path.join(meta_dir, "episodes.jsonl")
clean_episode_meta_lines = []

with open(episodes_meta_path, "r") as f:
    for line in f.readlines():
        data = json.loads(line)
        if data["episode_id"] not in delete_ep_ids:
            clean_episode_meta_lines.append(line)
        else:
            print(f"Removed meta entry: {data['episode_id']}")

# 4) 写回新的 meta 文件
with open(stats_path, "w") as f:
    f.writelines(clean_stats_lines)

with open(episodes_meta_path, "w") as f:
    f.writelines(clean_episode_meta_lines)

print("\nDONE. Clean dataset saved.")
print(f"Deleted {len(delete_ep_ids)} episodes.")

import sys
import random
import numpy as np
import os
from PIL import Image
from mujoco_env.y_env2 import SimpleEnv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# If you want to randomize the object positions, set this to None
# If you fix the seed, the object positions will be the same every time
SEED = 0 
# SEED = None <- Uncomment this line to randomize the object positions

REPO_NAME = 'omy_pnp_language'
NUM_DEMO = 50 # Number of demonstrations to collect
ROOT = "./demo_data_cup_ramdom" # The root directory to save the demonstrations

xml_path = './asset/example_scene_y2.xml'
# Define the environment
PnPEnv = SimpleEnv2(xml_path, seed = SEED, state_type = 'joint_angle')

create_new = True
if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("Do you want to delete it? (y/n) ")
    if ans == 'y':
        import shutil
        shutil.rmtree(ROOT)
    else:
        create_new = False


if create_new:
    dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root = ROOT, 
                robot_type="omy",
                fps=20, # 20 frames per second
                features={
                    "observation.image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "observation.wrist_image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["state"], # x, y, z, roll, pitch, yaw
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["action"], # 6 joint angles and 1 gripper
                    },
                    "obj_init": {
                        "dtype": "float32",
                        "shape": (9,),
                        "names": ["obj_init"], # just the initial position of the object. Not used in training.
                    },
                },
                image_writer_threads=10,
                image_writer_processes=5,
        )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)

action = np.zeros(7)
episode_id = 0
record_flag = False # Start recording when the robot starts moving
while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # check if the episode is done
        done = PnPEnv.check_success()
        if done: 
            # Save the episode data and reset the environment
            dataset.save_episode()
            PnPEnv.reset()
            episode_id += 1
        # Teleoperate the robot and get delta end-effector pose with gripper
        action, reset  = PnPEnv.teleop_robot()
        if not record_flag and sum(action) != 0:
            record_flag = True
            print("Start recording")
        if reset:
            # Reset the environment and clear the episode buffer
            # This can be done by pressing 'z' key
            # PnPEnv.reset(seed=SEED)
            PnPEnv.reset()
            dataset.clear_episode_buffer()
            record_flag = False
        # Step the environment
        # Get the end-effector pose and images
        agent_image,wrist_image = PnPEnv.grab_image()
        # # resize to 256x256
        agent_image = Image.fromarray(agent_image)
        wrist_image = Image.fromarray(wrist_image)
        agent_image = agent_image.resize((256, 256))
        wrist_image = wrist_image.resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)
        joint_q = PnPEnv.step(action)
        action = PnPEnv.q[:7] # 6 joint angles and 1 gripper
        action = action.astype(np.float32)
        if record_flag:
            # Add the frame to the dataset
            dataset.add_frame( {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": joint_q[:6], 
                    "action": action,
                    "obj_init": PnPEnv.obj_init_pose,
                    # "task": PnPEnv.instruction,
                }, task = PnPEnv.instruction
            )
        PnPEnv.render(teleop=True, idx=episode_id, clean_ui=False)


PnPEnv.env.close_viewer()

# Clean up the images folder
import shutil
shutil.rmtree(dataset.root / 'images')
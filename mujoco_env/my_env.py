import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw


class SimpleEnv:
    def __init__(self,
                 xml_path,
                 action_type='eef_pose',
                 state_type='joint_angle',
                 seed=None):
        """
        args:
            xml_path: str, path to the xml file
            action_type: str, type of action space, 'eef_pose','delta_joint_angle' or 'joint_angle'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose'
            seed: int, seed for random number generator
        """
        # Load the xml file
        self.env = MuJoCoParserClass(name='Tabletop', rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type

        self.joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'joint7',
        ]
        self.tcp_body = 'ee_body'          # 先用 body，和你现有 get_pR_body/solve_ik 兼容
        # Panda 两指关节（qpos 里是 0~0.04）
        self.gripper_joint = 'finger_joint1'  # 读一个就够（equality 约束会让两指相等）
        # Panda actuator8 的 ctrl 是 0~255（你 XML 里写死的）
        self.GRIP_OPEN  = 1
        self.GRIP_CLOSE = 0

        self.init_viewer()
        self.reset(seed)

    def init_viewer(self):
        """
        Initialize the viewer
        """
        self.env.reset()
        self.env.init_viewer(
            distance=2.0,
            elevation=-30,
            transparent=False,
            black_sky=True,
            use_rgb_overlay=False,
            loc_rgb_overlay='top right',
        )

    def reset(self, seed=None):
        """
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        """
        # if seed is not None:
        #     np.random.seed(seed=seed)

        # 1. 机械臂 IK 到一个固定初始位姿
        q_init = np.deg2rad([0, 0, 0, 0, 0, 0, 0])
        
        q_zero = np.array([0,-0.086, 0,-2.66065482,0,2.5,0.78], dtype=float)
        # q_zero, ik_err_stack, ik_info = solve_ik(
        #     env=self.env,
        #     joint_names_for_ik=self.joint_names,
        #     body_name_trgt=self.tcp_body,
        #     q_init=q_init,  # ik from zero pose
        #     p_trgt=np.array([0.3, 0, 1.0]), #位姿
        #     R_trgt=rpy2r(np.deg2rad([0, 0, 0])), #角度
        # )
        print("q_init(rad) =", q_init)
        self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)

        # 2. 固定盘子位置
        plate_xyz = np.array([0.4, -0.3, 0.82])
        self.env.set_p_base_body(body_name='body_obj_plate_11', p=plate_xyz)
        self.env.set_R_base_body(body_name='body_obj_plate_11', R=np.eye(3, 3))

        # === 定义两个区域（区域 A / 区域 B） ===
        region_A = dict(
            x_range=[0.4, 0.4],
            y_range=[-0, 0],
            z_range=[0.83, 0.83],
        )
        region_B = dict(
            x_range=[0.4, 0.4],
            y_range=[0.2, 0.2],
            z_range=[0.83, 0.83],
        )

        
        swap = 1#random.random() < 0.5

        if not swap:
            red_region = region_A
            blue_region = region_B
        else:
            red_region = region_B
            blue_region = region_A


        # Set object positions
        obj_xyzs = sample_xyzs(
            1,
            x_range=red_region["x_range"],
            y_range=red_region["y_range"],
            z_range=red_region["z_range"],
            min_dist  = 0.16,
            xy_margin = 0.0
        )
        self.env.set_p_base_body(body_name='body_obj_mug_5',p=obj_xyzs[0,:])
        self.env.set_R_base_body(body_name='body_obj_mug_5',R=np.eye(3,3))
        obj_xyzs = sample_xyzs(
            1,
            x_range=blue_region["x_range"],
            y_range=blue_region["y_range"],
            z_range=blue_region["z_range"],
            min_dist  = 0.16,
            xy_margin = 0.0
        )
        self.env.set_p_base_body(body_name='body_obj_mug_6',p=obj_xyzs[0,:])
        self.env.set_R_base_body(body_name='body_obj_mug_6',R=np.eye(3,3))
        self.env.forward(increase_tick=False)

        # 5. 保存初始状态
        self.last_q = copy.deepcopy(q_zero)
        self.q = np.concatenate([q_zero, np.array([0.0] * 1)])
        self.p0, self.R0 = self.env.get_pR_body(body_name=self.tcp_body)
        cube_red_init_pose, cube_blue_init_pose, plate_init_pose = self.get_obj_pose()
        # obj_init_pose: [red(6) + blue(6) + plate(6)] = 18 维
        self.obj_init_pose = np.concatenate(
            [cube_red_init_pose, cube_blue_init_pose, plate_init_pose],
            dtype=np.float32
        )

        # 仿真几步稳定一下
        for _ in range(100):
            self.step_env()

        # 随机生成语言指令 & 目标物体
        self.set_instruction()
        print("DONE INITIALIZATION")
        self.gripper_state = False
        self.past_chars = []

    def set_instruction(self, given = None):
        """
        Set the instruction for the task
        """
        if given is None:
            obj_candidates = ['red', 'blue']
            obj1 = random.choice(obj_candidates)
            self.instruction = f'Place the {obj1} mug on the plate.'
            if obj1 == 'red':
                self.obj_target = 'body_obj_mug_5'
            else:
                self.obj_target = 'body_obj_mug_6'
        else:
            self.instruction = given
            if 'red' in self.instruction:
                self.obj_target = 'body_obj_mug_5'
            elif 'blue' in self.instruction:
                self.obj_target = 'body_obj_mug_6'
            else:
                raise ValueError('Instruction does not contain a valid object color (red or blue).')


    def step(self, action):
        """
        Take a step in the environment
        """
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q, ik_err_stack, ik_info = solve_ik(
                env=self.env,
                joint_names_for_ik=self.joint_names,
                body_name_trgt=self.tcp_body,
                q_init=q,
                p_trgt=self.p0,
                R_trgt=self.R0,
                max_ik_tick=50,
                ik_stepsize=1.0,
                ik_eps=1e-2,
                ik_th=np.radians(5.0),
                render=False,
                verbose_warning=False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:-1] + self.last_q
        elif self.action_type == 'joint_angle':
            q = action[:-1]
        else:
            raise ValueError('action_type not recognized')

        grip = self.GRIP_OPEN if float(action[-1]) > 0.5 else self.GRIP_CLOSE   # 255 or 0
        gripper_cmd = np.array([grip], dtype=float)
        self.compute_q = q
        q = np.concatenate([q, gripper_cmd])

        self.q = q
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq = self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')

    def step_env(self):
        self.env.step(self.q)

    def grab_image(self):
        """
        grab images from the environment
        """
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name='agentview')
        self.rgb_ego = self.env.get_fixed_cam_rgb(
            cam_name='hand_cam')
        self.rgb_side = self.env.get_fixed_cam_rgb(
            cam_name='sideview')
        return self.rgb_agent, self.rgb_ego

    def render(self, teleop=False, idx=0):
        """
        Render the environment
        """
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name=self.tcp_body)
        R_current = R_current @ np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95, 0.05, 0.05, 0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05, 0.95, 0.05, 0.5])
        rgb_egocentric_view = add_title_to_img(self.rgb_ego, text='Egocentric View', shape=(640, 480))
        rgb_agent_view = add_title_to_img(self.rgb_agent, text='Agent View', shape=(640, 480))
        #self.env.plot_T(p=np.array([0.1, 0.0, 1.0]), label=f"Episode {idx}", plot_axis=False, plot_sphere=False)
        self.env.viewer_rgb_overlay(rgb_agent_view, loc='top right')
        self.env.viewer_rgb_overlay(rgb_egocentric_view, loc='bottom right')
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side, text='Side View', shape=(640, 480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
            self.env.viewer_text_overlay(text1='Key Pressed', text2='%s' % (self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated', text2='%s' % (self.env.get_key_repeated_list()))
        if getattr(self, 'instruction', None) is not None:
            language_instructions = self.instruction
            self.env.viewer_text_overlay(text1='Language Instructions', text2=language_instructions)
        self.env.render()

    def get_joint_state(self):
        """
        Get the joint state of the robot
        """
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper = self.env.get_qpos_joint(self.gripper_joint)
        gripper_cmd = 1.0 if gripper > 0.02 else 0.0
        return np.concatenate([qpos, [gripper_cmd]], dtype=np.float32)

    def teleop_robot(self):
        """
        Teleoperate the robot using keyboard
        """
        dpos = np.zeros(3)
        drot = np.eye(3)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
            dpos += np.array([0.007, 0.0, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
            dpos += np.array([-0.007, 0.0, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
            dpos += np.array([0.0, -0.007, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
            dpos += np.array([0.0, 0.007, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
            dpos += np.array([0.0, 0.0, 0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
            dpos += np.array([0.0, 0.0, -0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state = not self.gripper_state
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state], dtype=np.float32)], dtype=np.float32)
        return action, False

    def get_delta_q(self):
        """
        Get the delta joint angles of the robot
        """
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper = self.env.get_qpos_joint(self.gripper_joint)
        gripper_cmd = 1.0 if gripper > 0.02 else 0.0
        return np.concatenate([delta, [gripper_cmd]], dtype=np.float32)

    def check_success(self):
        """
        Check if the target cube (red or blue, according to instruction) is placed on the plate
        + Gripper should be open and move upward above 0.9
        """
        p_cube = self.env.get_p_body(self.obj_target)
        p_plate = self.env.get_p_body('body_obj_plate_11')
        if (np.linalg.norm(p_cube[:2] - p_plate[:2]) < 0.05 and
                np.linalg.norm(p_cube[2] - p_plate[2]) < 0.05 and
                self.env.get_qpos_joint(self.gripper_joint) > 0.02):
            p_tcp = self.env.get_p_body(self.tcp_body)[2]
            if p_tcp > 0.9:
                return True
        return False

    def get_obj_pose(self):
        """
        returns:
            p_cube_red: np.array, position of the red cube
            p_cube_blue: np.array, position of the blue cube
            p_plate: np.array, position of the plate
        """
        p_mug_red = self.env.get_p_body('body_obj_mug_5')
        p_mug_blue = self.env.get_p_body('body_obj_mug_6')
        p_plate = self.env.get_p_body('body_obj_plate_11')

        return p_mug_red, p_mug_blue, p_plate

    def set_obj_pose(self, p_mug_red, p_mug_blue, p_plate):
        '''
        Set the object poses
        args:
            p_mug_red: np.array, position of the red mug
            p_mug_blue: np.array, position of the blue mug
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name='body_obj_mug_5',p=p_mug_red)
        self.env.set_R_base_body(body_name='body_obj_mug_5',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_mug_6',p=p_mug_blue)
        self.env.set_R_base_body(body_name='body_obj_mug_6',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=p_plate)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        self.step_env()


    def get_ee_pose(self):
        """
        get the end effector pose of the robot + gripper state
        """
        p, R = self.env.get_pR_body(body_name=self.tcp_body)
        rpy = r2rpy(R)
        return np.concatenate([p, rpy], dtype=np.float32)

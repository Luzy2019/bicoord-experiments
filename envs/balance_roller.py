from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy


class balance_roller(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        ori_qpos = [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0, 0, 0.707, 0.707]]
        self.model_id = np.random.choice([0, 2], 1)[0]
        block_pos = rand_pose(
            xlim=[-0.1, 0.1],
            ylim=[-0.1, 0.1],
            rotate_rand=False,
        )
        rand_pos = rand_pose(
            xlim=[-0.15, 0.15],
            ylim=[-0.25, -0.15],
            qpos=ori_qpos[self.model_id],
        )
        self.roller = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="102_roller",
            convex=True,
            model_id=self.model_id,
        )
        self.block = create_box(
            scene=self,
            pose=block_pos,
            half_size=(0.025, 0.025, 0.025),
            color=np.random.rand(3),
        )

        self.add_prohibit_area(self.roller, padding=0.1)
        self.pick_check = 0

    def play_once(self):
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")

        self.move(
            self.grasp_actor(self.roller, left_arm_tag, pre_grasp_dis=0.08, contact_point_id=0),
            self.grasp_actor(self.roller, right_arm_tag, pre_grasp_dis=0.08, contact_point_id=1),
        )

        self.move(
            self.move_by_displacement(left_arm_tag, z=0.85 - self.roller.get_pose().p[2]),
            self.move_by_displacement(right_arm_tag, z=0.85 - self.roller.get_pose().p[2]),
        )

        delta = self.block.get_pose().p - self.roller.get_pose().p

        self.move(
            self.move_by_displacement(left_arm_tag, x=delta[0], y=delta[1], z=delta[2] + 0.025),
            self.move_by_displacement(right_arm_tag, x=delta[0], y=delta[1], z=delta[2] + 0.025),
        )

        self.move(
            self.open_gripper(left_arm_tag),
            self.open_gripper(right_arm_tag),
        )

        return self.info

    def check_success(self):
        roller_pose = self.roller.get_pose().p
        block_pose = self.block.get_pose().p
        eps = np.array([0.03, 0.03])
        if (self.is_left_gripper_close() and self.is_right_gripper_close() and roller_pose[2] > 0.8):
            self.pick_check = 1
        tag = np.all(abs(roller_pose[:2] - block_pose[:2]) < eps) and ((roller_pose[2] - block_pose[2]) > 0.015)
        self.stage_eval_score = (self.pick_check + int(tag)) / 2.0
        return tag


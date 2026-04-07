from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *


class place_plate_and_cup(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        GRAY = (0.5, 0.5, 0.5)

        rand_pos = rand_pose(
            xlim=[0.15, 0.16],
            ylim=[-0.15, -0.25],
            rotate_rand=False,
        )

        rand_pos_2 = rand_pose(
            xlim=[-0.13, -0.1],
            ylim=[-0.15, -0.25],
            rotate_rand=False,
        )

        rand_pos_3 = rand_pose(
            xlim=[-0.25, -0.23],
            ylim=[-0.05, -0.07],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )

        rand_pos_4 = rand_pose(
            xlim=[-0.08, -0.05],
            ylim=[0.12, 0.15],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )

        rand_pos_5 = rand_pose(
            xlim=[0.15, 0.25],
            ylim=[-0.05, -0.15],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )

        rand_pos_6 = rand_pose(
            xlim=[0.15, 0.25],
            ylim=[0.05, 0.15],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )

        self.target = create_box(
            scene=self,
            pose=rand_pos,
            half_size=(0.05, 0.05, 0.00005),
            color=GRAY,
            name="target",
            is_static=True,
        )

        self.target_2 = create_box(
            scene=self,
            pose=rand_pos_2,
            half_size=(0.05, 0.05, 0.00005),
            color=GRAY,
            name="target_2",
            is_static=True,
        )

        self.plate = create_actor(
            self,
            pose=rand_pos_3,
            modelname="003_plate",
            convex=True,
        )

        self.plate_2 = create_actor(
            self,
            pose=rand_pos_4,
            modelname="003_plate",
            convex=True,
        )

        cup_id = 5

        self.cup = create_actor(
            self,
            pose=rand_pos_5,
            modelname="021_cup",
            model_id=cup_id,
            convex=True,
        )

        self.cup_2 = create_actor(
            self,
            pose=rand_pos_6,
            modelname="021_cup",
            model_id=cup_id,
            convex=True,
        )

        self.add_prohibit_area(self.target, padding=0.01)
        self.add_prohibit_area(self.target_2, padding=0.01)
        self.add_prohibit_area(self.plate, padding=0.01)
        self.add_prohibit_area(self.plate_2, padding=0.01)
        self.add_prohibit_area(self.cup, padding=0.01)
        self.add_prohibit_area(self.cup_2, padding=0.01)

    def play_once(self):
        left = ArmTag("left")
        right = ArmTag("right")
        self.move(
            self.grasp_actor(self.plate, arm_tag=left, pre_grasp_dis=0.07, grasp_dis=0.0, contact_point_id=2),
            self.grasp_actor(self.cup, arm_tag=right, pre_grasp_dis=0.07, grasp_dis=0.0, contact_point_id=0)
        )

        self.move(
            self.move_by_displacement(right, z=0.04, move_axis="arm"),
            self.move_by_displacement(left, z=0.1, move_axis="arm")
        )

        self.move(
            self.place_actor(self.plate, target_pose=self.target.get_functional_point(0), arm_tag=left, functional_point_id=0, pre_dis=0.12, dis=0.03),
            self.place_actor(self.cup, target_pose=self.target.get_functional_point(0), arm_tag=right, functional_point_id=0, pre_dis=0.12, dis=0.03),
        )

        self.move(
            self.move_by_displacement(right, z=0.08, move_axis="arm"),
            self.move_by_displacement(left, z=0.08, move_axis="arm")
        )

        self.move(
            self.grasp_actor(self.plate_2, arm_tag=left, pre_grasp_dis=0.07, grasp_dis=0.0, contact_point_id=2),
            self.grasp_actor(self.cup_2, arm_tag=right, pre_grasp_dis=0.07, grasp_dis=0.0, contact_point_id=0)

        )

        self.move(
            self.move_by_displacement(right, z=0.04, move_axis="arm"),
            self.move_by_displacement(left, z=0.1, move_axis="arm")
        )

        self.move(
            self.place_actor(self.plate_2, target_pose=self.target_2.get_functional_point(0), arm_tag=left, functional_point_id=0, pre_dis=0.12, dis=0.03),
            self.place_actor(self.cup_2, target_pose=self.target_2.get_functional_point(0), arm_tag=right, functional_point_id=0, pre_dis=0.12, dis=0.03),
        )

        self.move(
            self.move_by_displacement(right, z=0.08, move_axis="arm"),
            self.move_by_displacement(left, z=0.08, move_axis="arm")
        )

        return self.info

    def check_success(self):
        target_pose = self.target.get_pose().p
        target_2_pose = self.target_2.get_pose().p
        plate_pose = self.plate.get_pose().p
        plate_2_pose = self.plate_2.get_pose().p
        cup_pose = self.cup.get_pose().p
        cup_2_pose = self.cup_2.get_pose().p
        eps = np.array([0.05, 0.05])
        if self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open():
            self.stage_eval_score = int(np.all(abs(plate_pose[:2] - target_pose[:2]) < eps)) + int(np.all(abs(plate_2_pose[:2] - target_2_pose[:2]) < eps)) + int(np.all(abs(cup_pose[:2] - target_pose[:2]) < eps)) + int(np.all(abs(cup_2_pose[:2] - target_2_pose[:2]) < eps))
            self.stage_eval_score = self.stage_eval_score / 4.0
        return (np.all(abs(plate_pose[:2] - target_pose[:2]) < eps) and np.all(abs(plate_2_pose[:2] - target_2_pose[:2]) < eps)
                and np.all(abs(cup_pose[:2] - target_pose[:2]) < eps) and np.all(abs(cup_2_pose[:2] - target_2_pose[:2]) < eps)
                and self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open())

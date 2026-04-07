from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *


class exchange_mics(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        GRAY = (0.5, 0.5, 0.5)
        rand_pos = rand_pose(
            xlim=[-0.2, -0.15],
            ylim=[-0.05, 0.0],
            qpos=[0.707, 0.707, 0, 0],
            rotate_rand=False,
        )
        while abs(rand_pos.p[0]) < 0.15:
            rand_pos = rand_pose(
                xlim=[-0.2, -0.15],
                ylim=[-0.05, 0.0],
                qpos=[0.707, 0.707, 0, 0],
                rotate_rand=False,
            )
        rand_pos_2 = rand_pose(
            xlim=[0.2, 0.25],
            ylim=[-0.05, 0.0],
            qpos=[0.707, 0.707, 0, 0],
            rotate_rand=False,
        )
        while abs(rand_pos.p[0]) < 0.15:
            rand_pos_2 = rand_pose(
                xlim=[0.2, 0.25],
                ylim=[-0.05, 0.0],
                qpos=[0.707, 0.707, 0, 0],
                rotate_rand=False,
            )

        target_pos = rand_pose(
            xlim=[rand_pos.p[0], rand_pos.p[0]],
            ylim=[rand_pos.p[1], rand_pos.p[1]],
            rotate_rand=False,
        )

        target_pos_2 = rand_pose(
            xlim=[rand_pos_2.p[0], rand_pos_2.p[0]],
            ylim=[rand_pos_2.p[1], rand_pos_2.p[1]],
            rotate_rand=False,
        )

        target_pos_3 = rand_pose(
            xlim=[0.15, 0.15],
            ylim=[-0.2, -0.2],
            rotate_rand=False,
        )

        self.microphone_id = np.random.choice([0, 4, 5], 1)[0]
        self.microphone_id_2 = np.random.choice([0, 4, 5], 1)[0]

        self.target = create_box(
            scene=self,
            pose=target_pos,
            half_size=(0.05, 0.05, 0.00005),
            color=GRAY,
            is_static=True,
        )

        self.target_2 = create_box(
            scene=self,
            pose=target_pos_2,
            half_size=(0.05, 0.05, 0.00005),
            color=GRAY,
            is_static=True,
        )

        self.microphone = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="018_microphone",
            convex=True,
            model_id=self.microphone_id,
        )

        self.microphone_2 = create_actor(
            scene=self,
            pose=rand_pos_2,
            modelname="018_microphone",
            convex=True,
            model_id=self.microphone_id_2,
        )

        self.handover_middle_pose = [0, -0.05, 0.85, 0, 1, 0, 0]

    def play_once(self):
        left = ArmTag("left")
        right = ArmTag("right")



        self.move(
            self.grasp_actor(
                self.microphone,
                arm_tag=left,
                contact_point_id=[1, 9, 10, 11, 12, 13, 14, 15],
                pre_grasp_dis=0.1,
            ),
            self.grasp_actor(
                self.microphone_2,
                arm_tag=right,
                contact_point_id=[1, 9, 10, 11, 12, 13, 14, 15],
                pre_grasp_dis=0.1,
            ),
        )

        right_pos = self.get_arm_pose("right")
        left_pos = self.get_arm_pose("left")

        self.move(
            self.move_by_displacement(right, x=-0.2),
        )

        self.move(
            self.open_gripper(right),
        )

        self.move(
           self.move_by_displacement(right, z=0.1),
        )

        self.move(
            self.move_by_displacement(
                left,
                z=0.12,
                quat=(GRASP_DIRECTION_DIC["front_right"]),
                move_axis="arm",
            ),
            self.back_to_origin(right)
        )

        self.move(
            self.place_actor(
                self.microphone,
                arm_tag=left,
                target_pose=self.handover_middle_pose,
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
                constrain="free",
            ),
            self.move_by_displacement(right, z=0.1)
        )

        self.move(
            self.grasp_actor(
                self.microphone,
                arm_tag=right,
                contact_point_id=[0, 2, 3, 4, 5, 6, 7, 8],
                pre_grasp_dis=0.1,
            ))

        self.move(
            self.open_gripper(left),
        )

        self.move(
            self.move_by_displacement(left, y=-0.05),
        )

        self.move(
            self.back_to_origin(left),
        )

        self.move(
            self.grasp_actor(
                self.microphone_2,
                arm_tag=left,
                contact_point_id=[1, 9, 10, 11, 12, 13, 14, 15],
                pre_grasp_dis=0.1,
            ),
            self.move_to_pose(right, right_pos),
        )

        self.move(
            self.move_to_pose(left, left_pos),
            self.move_by_displacement(right, y=0.05),
        )

        self.move(
            self.open_gripper(left),
            self.open_gripper(right),
        )

        return self.info

    def check_success(self):
        target_pos = self.target.get_pose().p
        target_pos_2 = self.target_2.get_pose().p
        microphone_pos = self.microphone.get_pose().p
        microphone_pos_2 = self.microphone_2.get_pose().p
        eps = np.array([0.05, 0.05, 0.05])
        tag = np.all(abs(microphone_pos - target_pos_2) < eps)
        tag_2 = np.all(abs(microphone_pos_2 - target_pos) < eps)

        self.stage_eval_score = (int(tag) + int(tag_2)) / 2.0
        return tag and tag_2 
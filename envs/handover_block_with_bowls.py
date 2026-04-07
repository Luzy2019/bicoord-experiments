from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *


class handover_block_with_bowls(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        GRAY = (0.5, 0.5, 0.5)

        rand_pos = rand_pose(
            xlim=[-0.20, -0.15],
            ylim=[-0.05, 0.05],
            rotate_rand=False,
        )

        rand_pos_2 = rand_pose(
            xlim=[0.15, 0.20],
            ylim=[-0.05, 0.05],
            rotate_rand=False,
        )

        rand_pos_3 = rand_pose(
            xlim=[-0.20, -0.15],
            ylim=[-0.15, -0.05],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )

        rand_pos_4 = rand_pose(
            xlim=[0.15, 0.20],
            ylim=[-0.15, -0.05],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )

        rand_pos_5 = rand_pose(
            xlim=[rand_pos_3.p[0], rand_pos_3.p[0]],
            ylim=[rand_pos_3.p[1], rand_pos_3.p[1]],
            zlim=[rand_pos_3.p[2]+0.01, rand_pos_3.p[2]+0.01],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )

        self.target_2 = create_box(
            scene=self,
            pose=rand_pos_2,
            half_size=(0.05, 0.05, 0.00005),
            color=GRAY,
            name="target_2",
            is_static=True,
        )

        cup_id = 5

        self.cup = create_actor(
            self,
            pose=rand_pos_3,
            modelname="002_bowl",
            model_id=cup_id,
            convex=True,
        )

        self.cup_2 = create_actor(
            self,
            pose=rand_pos_4,
            modelname="002_bowl",
            model_id=cup_id,
            convex=True,
        )

        self.block = create_box(
            scene=self,
            pose=rand_pos_5,
            half_size=(0.01, 0.01, 0.01),
            color=(1,0,0)
        )

        self.add_prohibit_area(self.target_2, padding=0.01)
        self.add_prohibit_area(self.cup, padding=0.01)
        self.add_prohibit_area(self.cup_2, padding=0.01)

    def play_once(self):
        left = ArmTag("left")
        right = ArmTag("right")

        self.move(
            self.grasp_actor(self.cup, arm_tag=left, pre_grasp_dis=0.07, grasp_dis=0.0, contact_point_id=2),
            self.grasp_actor(self.cup_2, arm_tag=right, pre_grasp_dis=0.07, grasp_dis=0.0, contact_point_id=0)
        )

        self.move(
            self.move_to_pose(left, [-0.09947217255830765, -0.1375633031129837, 1.0779637098312378, 0.6465598678427064, 0.27378176343060595, -0.26739928990963735, 0.6599253768903507]),
            self.move_to_pose(right, [0.06718142330646515, -0.06545648872852325, 1.0778271198272705, 0.49518559869556944, -0.5052240087047549, 0.49571178857877984, 0.5037953419165712]),        

        )

        self.move(
            self.place_actor(self.cup_2, target_pose=self.target_2.get_functional_point(0), arm_tag=right, functional_point_id=0, pre_dis=0.12, dis=0.03),
        )




        return self.info

    def check_success(self):
        target_2_pose = self.target_2.get_pose().p
        cup_pose = self.cup_2.get_pose().p
        block_pose = self.block.get_pose().p
        eps = np.array([0.025, 0.025, 0.025])

        self.stage_eval_score = (int(np.all(abs(cup_pose - target_2_pose) < eps)) + int(np.all(abs(block_pose - cup_pose) < eps))) / 2.0
        return np.all(abs(cup_pose - target_2_pose) < eps) and np.all(abs(block_pose - cup_pose) < eps)

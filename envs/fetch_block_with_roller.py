from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *


class fetch_block_with_roller(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):

        BLACK = np.array([0.0, 0.0, 0.0])
        l = 0.075
        w = 0.025
        h = 0.03
        x = -0.075
        y = 0
        z = 0.741 + h

        delta_y = 0.05

        down_pos = rand_pose(
            xlim=[x, x],
            ylim=[y-delta_y, y-delta_y],
            zlim=[z, z],
            rotate_rand=False,
        )

        up_pos = rand_pose(
            xlim=[x, x],
            ylim=[y+delta_y, y+delta_y],
            zlim=[z, z],
            rotate_rand=False,
        )

        middle_pos = rand_pose(
            xlim=[x, x],
            ylim=[y, y],
            zlim=[z+2*h, z+2*h],
            rotate_rand=False,
        )

        block_pos = rand_pose(
            xlim=[x-0.05, x+0.05],
            ylim=[y, y],
            rotate_rand=False,
        )


        self.down = create_box(
            scene=self,
            pose=down_pos,
            half_size=(l, w, h),
            color=BLACK,
            is_static=True,
        )

        self.up = create_box(
            scene=self,
            pose=up_pos,
            half_size=(l, w, h),
            color=BLACK,
            is_static=True,
        )

        self.middle = create_box(
            scene=self,
            pose=middle_pos,
            half_size=(l, 2*w+delta_y, 0.00005),
            color=BLACK,
            is_static=True,
        )


        self.block = create_box(
            scene=self,
            pose=block_pos,
            half_size=(0.025, 0.025, 0.025),
            color=np.random.rand(3),
        )

        roller_pos = rand_pose(
            xlim=[0.15, 0.15],
            ylim=[y, y],
            qpos=[0, 0, 0.707, 0.707],
            rotate_rand=False,
        )

        self.roller = create_actor(
            scene=self,
            pose=roller_pos,
            modelname="102_roller",
            convex=True,
            model_id=2,
        )

        self.out_check = 0


    def play_once(self):
        left = ArmTag("left")
        right = ArmTag("right")
        self.move(
            self.move_by_displacement(left, x=0.05, y=0.05, z=0.05),
            self.grasp_actor(self.roller, right, pre_grasp_dis=0.08, contact_point_id=1),
        )

        self.move(
            self.move_by_displacement(right, x=-0.25),
        )

        self.move(
            self.grasp_actor(self.block, arm_tag=left),
            self.move_by_displacement(right, x=0.25),
        )

        self.move(
            self.move_by_displacement(left, z=0.15),
        )

        self.move(
            self.place_actor(
                self.block,
                target_pose=self.middle.get_functional_point(0),
                arm_tag=left,
                functional_point_id=0, 
                pre_dis=0.12, 
                dis=0.03
            )
        )

        return self.info




    def check_success(self):
        middle_p = self.middle.get_pose().p
        block_p = self.block.get_pose().p
        eps = np.array([0.01, 0.01, 0.03])

        if block_p[0] < -0.15:
            self.out_check = 1
        self.stage_eval_score = (self.out_check + int(np.all(abs(middle_p - block_p) < eps))) / 2.0
        return np.all(abs(middle_p - block_p) < eps)
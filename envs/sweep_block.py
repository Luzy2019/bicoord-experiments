from ._base_task import Base_Task
from .utils import *
import sapien
from copy import deepcopy


class sweep_block(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(table_xy_bias=[0.3, 0], **kwags)

    def load_actors(self):

        GRAY = (1, 0, 0)
        dust_size = 0.02

        table_pose = rand_pose(
            xlim=[-0.055, -0.05],
            ylim=[0.0, 0.1],
        )

        self.table = create_box(
            scene=self,
            pose=table_pose,
            half_size=(0.05, 0.05, 0.05),
            color=(1,1,1),
            is_static=True
        )

        shovel_pose = rand_pose(
            xlim=[-0.3, -0.2],
            ylim=[-0.2, 0.0],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )

        brush_pose = rand_pose(
            xlim=[0.2, 0.25],
            ylim=[-0.1, 0.0],
            rotate_rand=False,
            qpos=[0, 0.707, -0.707, 0],
        )

        delta = 0.00
        table_x = table_pose.p[0]
        table_y = table_pose.p[1]
        table_z = table_pose.p[2]

        dust_pose = rand_pose(
            xlim=[table_x - delta, table_x + delta * 0],
            ylim=[table_y - delta * 0.5, table_y + delta * 0],
            zlim=[table_z + 0.02, table_z + 0.02]
        )

        self.shovel = create_actor(
            self.scene,
            pose=shovel_pose,
            modelname="082_smallshovel",
            model_id=3,
            convex=True,
        )

        self.brush = create_actor(
            self.scene,
            pose=brush_pose,
            modelname="083_brush",
            model_id=2,
            convex=True,
        )

        self.dust = create_box(
            scene=self,
            pose=dust_pose,
            half_size=(dust_size, dust_size, dust_size),
            color=GRAY,
        )

        self.dustbin = create_actor(
            self.scene,
            pose=sapien.Pose([-0.45, 0, 0], [0.5, 0.5, 0.5, 0.5]),
            modelname="011_dustbin",
            convex=True,
            is_static=True,
        )
        self.delay(2)
        self.shovel_check = 0

        self.dust.set_mass(0.0001)
        self.brush.set_mass(0.01)
        self.shovel.set_mass(0.01)

    def play_once(self):

        sweep_arm = ArmTag("right")
        shovel_arm = ArmTag("left")
        sweep_arm_pose = self.get_arm_pose(sweep_arm)
        sweep_arm_pose[:2] = self.brush.get_pose().p[:2] + np.array([-0.1, -0.1])
        sweep_arm_pose[2] -= 0.05
        sweep_arm_pose[3:] = np.array([0.5, -0.5, 0.5, 0.5])
        self.move(
            self.move_to_pose(sweep_arm, sweep_arm_pose),
            self.grasp_actor(self.shovel, arm_tag=shovel_arm, pre_grasp_dis=0.05, grasp_dis=0.0, contact_point_id=0),
        )
        self.move(self.close_gripper(sweep_arm))
        shovel_arm_pose = self.get_arm_pose(shovel_arm)
        shovel_pose = self.shovel.get_pose().p
        table_pose = self.table.get_pose().p
        shovel_arm_pose[:3] = shovel_arm_pose[:3] + table_pose - shovel_pose + np.array([-0.18, 0.0, 0.0])
        self.move(
            self.move_by_displacement(sweep_arm, z=0.15),
            self.move_to_pose(shovel_arm, shovel_arm_pose),
        )
        sweep_arm_pose = self.get_arm_pose(sweep_arm)
        sweep_arm_pose[3:] = np.array([0, 0, 0.707, 0.707])
        self.move(self.move_to_pose(sweep_arm, sweep_arm_pose))
        sweep_arm_pose = self.get_arm_pose(sweep_arm)
        sweep_arm_pose[:3] = table_pose + np.array([0.25, -0.1, 0.18])
        self.move(self.move_to_pose(sweep_arm, sweep_arm_pose))
        self.move(self.move_by_displacement(sweep_arm, x=-0.2))
        self.move(
            self.move_by_displacement(sweep_arm, x=0.2),
            self.move_by_displacement(shovel_arm, x=-0.2, y=-0.05)
            )
        self.move(
            self.move_by_displacement(shovel_arm, quat=[0,0,1,0]),
            )

        return self.info

    def check_target(self, dust_pose, target_pose=np.array([-0.45, 0]), eps = np.array([0.221, 0.325])):
        return (np.all(np.abs(dust_pose[:2] - target_pose) < eps) and dust_pose[2] > 0.2 and dust_pose[2] < 0.7)

    def check_success(self):
        dust_pose = self.dust.get_pose().p
        if self.check_target(dust_pose):
            self.stage_eval_score = 1.0
            return True
        else:
            self.stage_eval_score = 0.0
            return False

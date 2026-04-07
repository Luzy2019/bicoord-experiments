from ._base_task import Base_Task
from .utils import *
import sapien
import math

class collect_pens(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        pen_id = 0
        cup_id = 1

        pen_cup_pose = rand_pose(
            xlim=[-0.06, -0.06],
            ylim=[-0.05, -0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            ylim_prop=True,
            rotate_rand=False,
        )

        self.center = pen_cup_pose.p[:2]

        def get_pen_pose(xlim, ylim):
            qposes = [
                [-0.434997, 0.459021, -0.726497, -0.26885],
                [0.700121, 0.278818, -0.362417, 0.548402],
                [-0.666012, 0.305012, -0.338696, -0.590492],
                [0.249712, -0.554061, 0.775036, 0.173149],
                [0.800523, -0.0972921, 0.0987112, 0.583055],
                [0.71097, 0.274597, -0.380225, 0.523973],
                [-0.282287, 0.609697, -0.700071, -0.241837],
                [-0.711305, -0.272698, 0.373563, -0.529275],
            ]  
            while True:
                a = rand_pose(
                    xlim=xlim,
                    ylim=[0.02, 0.07],
                    qpos=qposes[np.random.randint(len(qposes))],
                    rotate_rand=False,
                )
                b = rand_pose(
                    xlim=xlim,
                    ylim=[-0.10, -0.05],
                    qpos=qposes[np.random.randint(len(qposes))],
                    rotate_rand=False,
                )
                if abs(a.p[1] - b.p[1]) > 0.08:
                    return a, b
        a, b = get_pen_pose([-0.2, -0.15], [-0.12, 0.1])
        c, d = get_pen_pose([0.23, 0.25], [-0.12, 0.1])
        pen_pose_lst = [a, b, c, d]

        pen_pose_lst = sorted(pen_pose_lst, key=lambda x: x.p[0])

        def create_pen(pen_pose, id):
            return create_actor(self, pose=pen_pose, modelname="058_markpen", model_id=id, convex=True)

        self.pen1 = create_pen(pen_pose_lst[0], pen_id)
        self.pen2 = create_pen(pen_pose_lst[1], pen_id)
        self.pen3 = create_pen(pen_pose_lst[2], pen_id)
        self.pen4 = create_pen(pen_pose_lst[3], pen_id)


        self.add_prohibit_area(self.pen1, padding=0.07)
        self.add_prohibit_area(self.pen2, padding=0.07)
        self.add_prohibit_area(self.pen3, padding=0.07)
        self.add_prohibit_area(self.pen4, padding=0.07)
        self.cup = create_actor(self, pose=pen_cup_pose, modelname="059_pencup_jlk", model_id=cup_id, convex=True, is_static=True) # 1 5
        self.add_prohibit_area(self.cup, padding=0.07)


        self.target_pose_left = [-0.16+self.center[0], self.center[1], 1.05, 0, 1, 0, 0]
        self.target_pose_right = [0.16+self.center[0], self.center[1], 1.05, 0, 0, 0, 1]

    def play_once(self):
        def get_grab_pos(pen, arm_tag):
            pose = pen.get_pose()
            pose.p += np.array([-0.07, 0, 0.18])
            if arm_tag == "left":
                pose.q = [0, -0.707, 0, 0.707] 
            else:
                pose.q = [0.707, 0, 0.707, 0]
            return pose

        lift_height = 0.1
        down_height = 0.05
        arm_tag = ArmTag('left')
        move_list_a = [self.pen1, self.pen2]
        move_list_b = [self.pen3, self.pen4]
        target_pose_a = self.target_pose_left
        target_pose_b = self.target_pose_right
        f = 1

        if np.random.rand() < 0.5:
            arm_tag = arm_tag.opposite
            move_list_a, move_list_b = move_list_b, move_list_a
            target_pose_a, target_pose_b = target_pose_b, target_pose_a
            f = -1
        if np.random.rand() < 0.5:
            move_list_a = move_list_a[::-1]
        if np.random.rand() < 0.5:
            move_list_b = move_list_b[::-1]

        pose_a = get_grab_pos(move_list_a[0], arm_tag)
        pose_b = get_grab_pos(move_list_b[0], arm_tag.opposite)

        self.move(
            self.move_to_pose(arm_tag, pose_a),
            self.move_to_pose(arm_tag.opposite, pose_b),
        )
        self.move(
            self.move_by_displacement(arm_tag, z=-down_height),
            self.move_by_displacement(arm_tag.opposite, z=-down_height),
        )
        self.move(
            self.close_gripper(arm_tag),
            self.close_gripper(arm_tag.opposite),
        )
        self.move(
            self.move_by_displacement(arm_tag, z=lift_height),
            self.move_by_displacement(arm_tag.opposite, z=lift_height),
        )
        self.move(
            self.move_to_pose(arm_tag, target_pose_a)
        )
        self.move(
            self.open_gripper(arm_tag)
        )
        

        pose_a = get_grab_pos(move_list_a[1], arm_tag)
        self.move(
            self.move_by_displacement(arm_tag, x=-f*0.15),
            self.move_to_pose(arm_tag.opposite, target_pose_b),
        )
        self.move(
            self.move_to_pose(arm_tag, pose_a),
            self.open_gripper(arm_tag.opposite)
        )
        
        pose_b = get_grab_pos(move_list_b[1], arm_tag.opposite)
        self.move(
            self.move_by_displacement(arm_tag, z=-down_height),
            self.move_by_displacement(arm_tag.opposite, x=f*0.15),
        )
        self.move(
            self.close_gripper(arm_tag),
            self.move_to_pose(arm_tag.opposite, pose_b),
        )
        self.move(
            self.move_by_displacement(arm_tag, z=lift_height),
            self.move_by_displacement(arm_tag.opposite, z=-down_height),
        )
        self.move(
            self.move_to_pose(arm_tag, target_pose_a),
            self.close_gripper(arm_tag.opposite)
        )
        self.move(
            self.open_gripper(arm_tag),
            self.move_by_displacement(arm_tag.opposite, z=lift_height),
        )

        self.move(
            self.back_to_origin(arm_tag),
            self.move_to_pose(arm_tag.opposite, target_pose_b)
        )
        self.move(
            self.open_gripper(arm_tag.opposite)
        )

        return self.info

    def check_success(self):
        poses = [self.pen1.get_pose().p, self.pen2.get_pose().p, self.pen3.get_pose().p, self.pen4.get_pose().p]

        eps1 = 0.07
        eps2 = 0.12

        cnt = 0
        for i in range(0, 4):
            if np.all(abs(poses[i][:2] - np.array([self.center[0], self.center[1]])) <= eps1) or \
            (np.all(abs(poses[i][:2] - np.array([self.center[0], self.center[1]])) <= eps2) and poses[i][2] > 0.8):
                cnt += 1
        if not self.is_left_gripper_open() or not self.is_right_gripper_open():
            cnt = min(cnt, 3)
        
        self.stage_eval_score = cnt / 4.0
        return cnt == 4
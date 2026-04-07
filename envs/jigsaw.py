from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *


class jigsaw(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):

        color = np.random.rand(3)
        color_size = [0.020, 0.020, 0.020]
        white = np.array([1.0, 1.0, 1.0])
        white_size = [0.020, 0.020, 0.00005]

        delta_x = 0.0
        delta_y = 0.1

        rand_pos_1 = rand_pose(
            xlim=[-0.10 - delta_x, -0.10 - delta_x],
            ylim=[0.10 - delta_y, 0.10 - delta_y],
            rotate_rand=False,
        )

        rand_pos_2 = rand_pose(
            xlim=[-0.10 - delta_x, -0.10 - delta_x],
            ylim=[0 - delta_y, 0 - delta_y],
            rotate_rand=False,
        )

        rand_pos_3 = rand_pose(
            xlim=[-0.10 - delta_x, -0.10 - delta_x],
            ylim=[-0.10 - delta_y, -0.10 - delta_y],
            rotate_rand=False,
        )

        rand_pos_4 = rand_pose(
            xlim=[0 - delta_x, 0 - delta_x],
            ylim=[0.10 - delta_y, 0.10 - delta_y],    
            rotate_rand=False,
        )

        rand_pos_5 = rand_pose(
            xlim=[0 - delta_x, 0 - delta_x],
            ylim=[0 - delta_y, 0 - delta_y],          
            rotate_rand=False,
        )

        rand_pos_6 = rand_pose(
            xlim=[0 - delta_x, 0 - delta_x],
            ylim=[-0.10 - delta_y, -0.10 - delta_y],  
            rotate_rand=False,
        )

        rand_pos_7 = rand_pose(
            xlim=[0.10 - delta_x, 0.10 - delta_x],    
            ylim=[0.10 - delta_y, 0.10 - delta_y],    
            rotate_rand=False,
        )

        rand_pos_8 = rand_pose(
            xlim=[0.10 - delta_x, 0.10 - delta_x],    
            ylim=[0 - delta_y, 0 - delta_y],          
            rotate_rand=False,
        )

        rand_pos_9 = rand_pose(
            xlim=[0.10 - delta_x, 0.10 - delta_x],    
            ylim=[-0.10 - delta_y, -0.10 - delta_y],  
            rotate_rand=False,
        )

        self.poses = [rand_pos_1, rand_pos_2, rand_pos_3,
                 rand_pos_4, rand_pos_5, rand_pos_6,
                 rand_pos_7, rand_pos_8, rand_pos_9]

        self.whites = []
        for i in range(9):
            self.whites.append(
                create_box(
                    scene=self,
                    pose=self.poses[i],
                    half_size=white_size,
                    color=white,
                    is_static=True,
                )
            )
        
        self.select = np.random.choice(range(9), 4, replace=False).tolist()

        self.blocks = []
        for i in range(9):
            if i in self.select:
                continue
            self.blocks.append(
                create_box(
                    scene=self,
                    pose=self.poses[i],
                    half_size=color_size,
                    color=color,
                )
            )

        left_1_pos = rand_pose(
            xlim=[-0.20 - delta_x, -0.20 - delta_x],  
            ylim=[-0.1 - delta_y, -0.1 - delta_y],    
            rotate_rand=False,
        )

        left_2_pos = rand_pose(
            xlim=[-0.20 - delta_x, -0.20 - delta_x],  
            ylim=[0.1 - delta_y, 0.1 - delta_y],      
            rotate_rand=False,
        )

        right_1_pos = rand_pose(
            xlim=[0.20 - delta_x, 0.20 - delta_x],    
            ylim=[-0.1 - delta_y, -0.1 - delta_y],    
            rotate_rand=False,
        )

        right_2_pos = rand_pose(
            xlim=[0.20 - delta_x, 0.20 - delta_x],    
            ylim=[0.1 - delta_y, 0.1 - delta_y],      
            rotate_rand=False,
        )

        self.select_blocks = []
        select_poses = [left_1_pos, left_2_pos, right_1_pos, right_2_pos]
        for i in range(4):
            self.select_blocks.append(
                create_box(
                    scene=self,
                    pose=select_poses[i],
                    half_size=color_size,
                    color=color,
                )
            )

        self.left_blocks = [self.select_blocks[0], self.select_blocks[1]]
        self.right_blocks = [self.select_blocks[2], self.select_blocks[3]]

    def get_grasp_targets(self, left_seq=[0,1,2,3,4,5,8,7,6], right_seq=[6,7,8,5,4,3,2,1,0]):
        left_target = None
        for num in left_seq:
            if num in self.select:
                left_target = num 
                self.select.remove(num)
                break
        
        right_target = None
        for num in right_seq:
            if num in self.select:
                right_target = num
                self.select.remove(num)
                break
        
        return left_target, right_target

    def play_once(self):
        self.copy_select = self.select.copy()
        self.record = []
        left = ArmTag("left")
        right = ArmTag("right")
        self.move(
            self.grasp_actor(self.left_blocks[1], arm_tag=left),
            self.grasp_actor(self.right_blocks[1], arm_tag=right)
        )
        self.move(
            self.move_by_displacement(left, z=0.12),
            self.move_by_displacement(right, z=0.12)
        )
        left_target, right_target = self.get_grasp_targets()

        self.move(
            self.place_actor(
                self.left_blocks[1],
                target_pose=self.whites[left_target].get_functional_point(0),
                arm_tag=left,
                functional_point_id=0, 
                pre_dis=0.12, 
                dis=0.03
            ),
        )
        self.move(
            self.move_by_displacement(left, z=0.12),
        )
        self.move(
            self.grasp_actor(self.left_blocks[0], arm_tag=left),
            self.place_actor(
                self.right_blocks[1],
                target_pose=self.whites[right_target].get_functional_point(0),
                arm_tag=right,
                functional_point_id=0, 
                pre_dis=0.12, 
                dis=0.03
            )
        )
        self.move(
            self.move_by_displacement(left, z=0.12),
            self.move_by_displacement(right, z=0.12),
        )
        left_target, right_target = self.get_grasp_targets()

        self.move(
            self.place_actor(
                self.left_blocks[0],
                target_pose=self.whites[left_target].get_functional_point(0),
                arm_tag=left,
                functional_point_id=0, 
                pre_dis=0.12, 
                dis=0.03
            ),
            self.grasp_actor(self.right_blocks[0], arm_tag=right)
        )

        self.move(
            self.move_by_displacement(left, z=0.12),
            self.move_by_displacement(right, z=0.12),
        )

        self.move(
            self.back_to_origin(left),
            self.place_actor(
                self.right_blocks[0],
                target_pose=self.whites[right_target].get_functional_point(0),
                arm_tag=right,
                functional_point_id=0, 
                pre_dis=0.12, 
                dis=0.03
            )
        )

        return self.info

    def check(self, p, eps=np.array([0.01, 0.01, 0.03])):
        for pos in self.poses:
            if np.all(abs(np.array(pos.p) - np.array(p)) < eps):
                return True
        return False



    def check_success(self):
        cnt = 9
        for block in self.select_blocks:
            block_pos = block.get_pose().p
            if not self.check(block_pos):
                cnt -= 1
        for block in self.blocks:
            block_pos = block.get_pose().p
            if not self.check(block_pos):
                cnt -= 1

        self.stage_eval_score = max(cnt - 5, 0) / 4.0
        return cnt == 9
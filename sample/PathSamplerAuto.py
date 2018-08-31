from os.path import *
from AirSimClient import *
import argparse
from math import sin, cos, pi
from ForestEnv import get_anchors
'''
The sampling process includes four levels:
  waypoints(anchor), round, sequence and images
  
1.  Users should drag the drone in the environment to
    collect enough anchors, and specify the anchor 
    position in ForestEnv.py, where the first three 
    dimension is [x, y, z] and the rest are the starting
    angles.

2.  Be careful that the z in [x, y, z] is the desired height
    you want the drone to fly to, not the ground point. Based
    on this z, there will be another random height to be added,
    tune this height in Controller, 
    called self.start_random_height

3.  Each time to run this script, please ensure the dataset
    folder is empty! 
    
    
4. Please ensure the drone is at least 1 meter high above the ground


How to use this script:

1. put AirSimClient.py and this file in the same folder

2. modify pnt_root path in main function
    pnt_root = join('/home/zhudelong/Videos/airsim', str(sample_idx))

3. drag the drone to desired position and orientation and record the points

4. run airsim, then this script

'''


class SaveData():
    def __init__(self, args, client):
        self.args = args
        self.idx = 0
        self.gimbal_ind = 0  # 0-front center 3-bottom
        self.state_client = client
        self.state_client.confirmConnection()
        self.state_client.enableApiControl(True)
        self.set_actor_segmentation()
        self.img_type = []
        self.image_request()
        self.start_loop = True

    def set_actor_segmentation(self):
        success = self.state_client.simSetSegmentationObjectID("[\w]*", 0, True)
        if not success:
            print('[\w]* failed!')
            return False
        success = self.state_client.simSetSegmentationObjectID('actor_car', 1, True)
        if not success:
            print('[\w]* failed!')
            return False
        return True

    def image_request(self):
        self.img_type.append(ImageRequest(0, AirSimImageType.Scene))
        self.img_type.append(ImageRequest(3, AirSimImageType.Scene))

        self.img_type.append(ImageRequest(0, AirSimImageType.DepthPlanner, True))
        self.img_type.append(ImageRequest(3, AirSimImageType.DepthPlanner, True))

        # self.img_type.append(ImageRequest(0, AirSimImageType.Segmentation))
        # self.img_type.append(ImageRequest(0, AirSimImageType.Segmentation))

    def save_kinematics(self, image_f, image_b, states, seq):
        kinematic = []
        kinematic.append(image_f.camera_position.x_val)
        kinematic.append(image_f.camera_position.y_val)
        kinematic.append(image_f.camera_position.z_val)

        kinematic.append(image_f.camera_orientation.x_val)
        kinematic.append(image_f.camera_orientation.y_val)
        kinematic.append(image_f.camera_orientation.z_val)
        kinematic.append(image_f.camera_orientation.w_val)

        kinematic.append(image_b.camera_position.x_val)
        kinematic.append(image_b.camera_position.y_val)
        kinematic.append(image_b.camera_position.z_val)

        kinematic.append(image_b.camera_orientation.x_val)
        kinematic.append(image_b.camera_orientation.y_val)
        kinematic.append(image_b.camera_orientation.z_val)
        kinematic.append(image_b.camera_orientation.w_val)

        kinematic.append(states.kinematics_estimated.angular_velocity.x_val)
        kinematic.append(states.kinematics_estimated.angular_velocity.y_val)
        kinematic.append(states.kinematics_estimated.angular_velocity.z_val)

        kinematic.append(states.kinematics_estimated.linear_acceleration.x_val)
        kinematic.append(states.kinematics_estimated.linear_acceleration.y_val)
        kinematic.append(states.kinematics_estimated.linear_acceleration.z_val)

        name = '0' * (6 - len(str(self.idx))) + str(self.idx)
        file_name = join(seq, self.args.poses, name)
        np.save(file_name, np.asarray(kinematic))

    def start(self):
        while (self.start_loop):
            self.loop()

    def loop(self, seq):
        idx = self.idx
        states = self.state_client.getMultirotorState()

        # actor_pos = self.state_client.simGetObjectPose('TargetPoint22')
        # print actor_pos
        images = self.state_client.simGetImages(self.img_type)
        # print int(states.timestamp / 1000000000)
        # print int(images[0].time_stamp / 1000000000)
        # print int(images[1].time_stamp / 1000000000)
        # print int(images[2].time_stamp / 1000000000)
        # print int(images[3].time_stamp / 1000000000)

        name = '0' * (6 - len(str(idx))) + str(idx)
        # save rgb image of front camera
        assert images[0].image_type == AirSimImageType.Scene
        file_name = join(seq, self.args.front_rgb, name + '.png')
        self.state_client.write_file(file_name, images[0].image_data_uint8)

        # save rgb image of bottom camera
        assert images[1].image_type == AirSimImageType.Scene
        file_name = join(seq, self.args.bottom_rgb, name + '.png')
        self.state_client.write_file(file_name, images[1].image_data_uint8)

        # save depth image of bottom camera
        assert images[2].image_type == AirSimImageType.DepthPlanner
        file_name = join(seq, self.args.front_depth, name)
        depth1 = self.state_client.listTo2DFloatArray(images[2].image_data_float,
                                                      images[2].width, images[2].height)
        np.save(file_name, depth1)
        # cv2.imwrite(file_name + '.png', depth.astype(np.uint8))
        # save depth image of bottom camera
        assert images[3].image_type == AirSimImageType.DepthPlanner
        file_name = join(seq, self.args.bottom_depth, name)
        depth2 = self.state_client.listTo2DFloatArray(images[3].image_data_float,
                                                      images[3].width, images[3].height)
        np.save(file_name, depth2)
        # cv2.imwrite(file_name + '.png', depth.astype(np.uint8))

        self.save_kinematics(images[0], images[1], states, seq)
        self.idx += 1



class Controller:
    def __init__(self, args, ip=''):
        self.args = args
        self.cmd_client = MultirotorClient(ip=ip)
        self.cmd_client.confirmConnection()
        self.cmd_client.enableApiControl(True)
        self.cmd_client.takeoff(2)
        self.pnt_idx = 0
        self.seq_idx = 0
        self.save_data = SaveData(args, self.cmd_client)

        # the drone speed range
        self.high_speed = 0.5
        self.low_speed = 0.1

        # the yaw rate range
        self.yaw_range = [1.0, 8.0]

        # represent different sampling mode
        self.mask_t = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

        # sampling time for each mode in self.mask_t
        self.tt_time = 8
        self.ty_time = 8

        # sampling time for yaw
        self.yy_time = 6
        self.yy_seq = 4

        # the random height for the starting of each round
        self.start_random_height = [-0.5, 1.0]


    def make_dir(self):
        pnt = join(self.args.root, str(self.pnt_idx))
        seq = join(pnt, str(self.seq_idx))
        if exists(seq):
            raise IOError('Folder Already Exist!')

        os.makedirs(join(seq, args.front_rgb))
        os.makedirs(join(seq, args.bottom_rgb))
        os.makedirs(join(seq, args.front_depth))
        os.makedirs(join(seq, args.bottom_depth))
        os.makedirs(join(seq, args.poses))

        return seq

    def write(self, interval):
        start = time.time()

        # get current sequence number and make corresponding dirs
        seq = self.make_dir()

        # remove start unstable frames
        time.sleep(1.5)
        print 'saving ...'

        # write data
        img_num = 0
        crs_num = 0
        while True:
            self.save_data.loop(seq)
            now = time.time()
            img_num += 1
            if now - start > interval - 1.5:
                break
        self.seq_idx += 1
        time.sleep(1.5)
        print 'finished saving, total {} images, crashed {} images!'.format(img_num, crs_num)

    def parse_vel(self, vel):
        roll_pitch_yaw = self.cmd_client.getPitchRollYaw()
        yaw = roll_pitch_yaw[2]
        x = vel[0] * cos(yaw) - vel[1] * sin(yaw)
        y = vel[0] * sin(yaw) + vel[1] * cos(yaw)
        return [x, y, vel[2]]



    def tt(self, mode, vel, interval):
        # to increase randomness, we randomly sample speed for each mod
        vel = np.random.uniform(self.low_speed, self.high_speed, size=[3, ])
        sign = np.random.randint(0, 2, 2)
        if sign[0] == 1:
            vel[0] = -vel[0]
        if sign[1] == 1:
            vel[1] = -vel[1]
        vel = self.parse_vel(vel)
        print "current mode: {}, {}, {}".format(mode[0] * vel[0], mode[1] * vel[1], mode[2] * vel[2])

        # forward
        self.cmd_client.moveByVelocity(mode[0] * vel[0], mode[1] * vel[1], -mode[2] * vel[2], duration=interval)
        self.write(interval)

        # backward
        self.cmd_client.moveByVelocity(-mode[0] * vel[0], -mode[1] * vel[1], mode[2] * vel[2], duration=interval)
        # self.write(interval)
        time.sleep(interval)

    def ty(self, mode, vel, yaw, interval):
        # increase randomness
        vel = np.random.uniform(self.low_speed, self.high_speed, size=[3, ])

        sign = np.random.randint(0, 2, 3)
        if sign[0] == 1:
            vel[0] = -vel[0]
        if sign[1] == 1:
            vel[1] = -vel[1]
        if sign[2] == 1:
            yaw = -yaw

        vel = self.parse_vel(vel)
        print "current mode: {}, {}, {}, {}".format(mode[0] * vel[0], mode[1] * vel[1], mode[2] * vel[2], yaw)

        # forward
        yaw_mode = YawMode(True, yaw)

        self.cmd_client.moveByVelocity(mode[0] * vel[0], mode[1] * vel[1], -mode[2] * vel[2], duration=interval, yaw_mode=yaw_mode)
        self.write(interval)

        # backward
        yaw_mode = YawMode(True, -yaw)
        self.cmd_client.moveByVelocity(-mode[0] * vel[0], -mode[1] * vel[1], mode[2] * vel[2], duration=interval, yaw_mode=yaw_mode)
        # self.write(interval)
        time.sleep(interval)

    def yy(self):
        for i in range(self.yy_seq):
            # increase randomness
            yaw = np.random.uniform(low=self.yaw_range[0], high=self.yaw_range[1])
            height = np.random.uniform(low = 0.3, high=1.0)
            print "current mode: {}".format(yaw)

            # sample yaw
            self.cmd_client.rotateByYawRate(yaw, self.yy_time)
            self.write(self.yy_time)

            # random height
            self.cmd_client.moveByVelocity(0.0, 0.0, -height, duration=1)
            time.sleep(1)

    def execute_single_path(self):
        # todo: remove this sentence and modify the tt and ty
        vel = np.random.uniform(self.low_speed, self.high_speed, size=[3, ])
        vel = self.parse_vel(vel)

        # tt combination
        for mode in self.mask_t:
            self.tt(mode, vel, self.tt_time)

        # ty combination
        for mode in self.mask_t:
            yaw = np.random.uniform(low=self.yaw_range[0], high=self.yaw_range[1])
            self.ty(mode, vel, yaw, self.ty_time)

        # yy
        self.yy()

    def execute_sampling(self, num, yaws):
        # record the sample location
        cur_pos = self.cmd_client.getPosition()
        print('current position is : {}, {}, {}'.format(cur_pos.x_val, cur_pos.y_val, cur_pos.z_val))

        for i in range(num):

            # reset sequence idx for each sample
            self.seq_idx = 0
            print 'start {}-th sampling at current position!'.format(i)

            # rotate to sample yaws
            self.cmd_client.rotateToYaw(yaws[i])
            time.sleep(1)
            print 'yaw angle is set to {}!'.format(yaws[i])

            # randomly start height
            height = np.random.uniform(self.start_random_height[0], self.start_random_height[1])
            self.cmd_client.moveByVelocity(0.0, 0.0, -height, duration=1)
            time.sleep(1)
            cur_poses = self.cmd_client.getPosition()
            print('current position after random height is : {}, {}, {}'.format(cur_poses.x_val, cur_poses.y_val, cur_poses.z_val))

            # sample i-th data at this location
            self.execute_single_path()

            # reset the sample position
            self.cmd_client.moveToPosition(cur_pos.x_val, cur_pos.y_val, cur_pos.z_val, 0.5)
            print 'reset {}-th sampling location finished!'.format(i)

            # go to next round at this location
            self.pnt_idx += 1

        self.pnt_idx = 0



def make_dir(args, root):
    if os.path.exists(root):
        raise IOError('Folder Already Exist!')
    os.mkdir(root)
    os.mkdir(join(root, args.front_rgb))
    os.mkdir(join(root, args.bottom_rgb))
    os.mkdir(join(root, args.front_depth))
    os.mkdir(join(root, args.bottom_depth))
    os.mkdir(join(root, args.poses))
    print('make dir:{}'.format(root))
    return root

# 45 -- first cross
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.front_rgb = '1.front_rgb'
    args.bottom_rgb = '2.bottom_rgb'
    args.front_depth = '3.front_depth'
    args.bottom_depth = '4.bottom_depth'
    args.poses = '5.poses'
    args.root = '/home/zhudelong/Dataset'

    ctrl = Controller(args)
    anchors = get_anchors()
    anchor_num = anchors.shape[0]
    print anchor_num
    round_num = anchors.shape[1] - 3

    for idx in range(anchor_num):
        print('+++++this is {} anchor!'.format(idx))

        # for avoiding collision
        if idx == 57:
            ctrl.cmd_client.moveToPosition(anchors[11, 0], anchors[11, 1], anchors[11, 2], velocity=3)

        # move to that target
        print('move to ancher: {}, {}, {}'.format(anchors[idx, 0], anchors[idx, 1], anchors[idx, 2]))
        ctrl.cmd_client.moveToPosition(anchors[idx, 0], anchors[idx, 1], anchors[idx, 2], velocity=3)
        time.sleep(2)

        # reset the rotation
        ctrl.cmd_client.rotateToYaw(0)
        time.sleep(2)

        # make directories
        pnt_root = join('/home/zhudelong/Dataset/airsim_data', str(idx))

        if exists(pnt_root):
            raise IOError('{} exist!'.format(pnt_root))
        os.mkdir(pnt_root)

        ctrl.args.root = pnt_root

        # sample data for sample_num round
        yaws = anchors[idx, 3:anchors.shape[1]]
        ctrl.execute_sampling(round_num, yaws)

    ctrl.cmd_client.reset()

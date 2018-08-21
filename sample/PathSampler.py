from os.path import *
from AirSimClient import *
import argparse

'''
How to use this script:
1. put AirSimClient.py and this file in the same folder

2. modify root path in main function
    root = join('/home/zhudelong/Videos/airsim', str(sample_idx))

3. modify the sample_idx in main function for "each new sampling location"
    sample_idx = 0

4. drag the drone to desired position

5. run airsim, then this script

6. after sampling one round, please slack Deron for checking
     
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
        self.high_speed = 0.7
        self.low_speed = 0.1
        self.mask_t = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

        self.target = np.asarray([[7143, -557, 409],
                                  [5769, -1679, 530],
                                  [4828, -3595, 685],
                                  [6919, -4799, 0.00],
                                  [6787, 1622, -500]]) / 100
        self.orig = np.array([5650, 730, 380]) / 100.0
        self.tgt = self.target - self.orig
        self.tgt[:, 2] = -self.tgt[:, 2]


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
        seq = self.make_dir()
        time.sleep(1)
        print 'saving ...'
        while True:
            self.save_data.loop(seq)
            now = time.time()
            if now - start > interval - 1:
                break
        self.seq_idx += 1
        time.sleep(1)
        print 'finished saving!'

    def tt(self, mode, vel, interval):
        # to increase randomness, we randomly sample speed for each mod
        vel = np.random.uniform(self.low_speed, self.high_speed, size=[3, ])
        print "current mode: {}, {}, {}".format(mode[0] * vel[0], mode[1] * vel[1], mode[2] * vel[2])

        # forward
        self.cmd_client.moveByVelocity(mode[0] * vel[0], mode[1] * vel[1], -mode[2] * vel[2], duration=interval)
        self.write(interval)

        # backward
        self.cmd_client.moveByVelocity(-mode[0] * vel[0], -mode[1] * vel[1], mode[2] * vel[2], duration=interval)
        self.write(interval)

    def ty(self, mode, vel, yaw, interval):
        # increase randomness
        vel = np.random.uniform(self.low_speed, self.high_speed, size=[3, ])
        print "current mode: {}, {}, {}, {}".format(mode[0] * vel[0], mode[1] * vel[1], mode[2] * vel[2], yaw)

        # forward
        yaw_mode = YawMode(True, yaw)
        self.cmd_client.moveByVelocity(mode[0] * vel[0], mode[1] * vel[1], -mode[2] * vel[2], duration=interval, yaw_mode=yaw_mode)
        self.write(interval)

        # backward
        yaw_mode = YawMode(True, -yaw)
        self.cmd_client.moveByVelocity(-mode[0] * vel[0], -mode[1] * vel[1], mode[2] * vel[2], duration=interval, yaw_mode=yaw_mode)
        self.write(interval)

    def yy(self):
        for i in range(6):
            # increase randomness
            yaw = np.random.uniform(low=0.5, high=8.0)
            height = np.random.uniform(low=0.5, high=1)
            print "current mode: {}".format(yaw)

            # sample yaw
            self.cmd_client.rotateByYawRate(yaw, 6)
            self.write(6)

            # random height
            self.cmd_client.moveByVelocity(0.0, 0.0, -height, duration=1)
            time.sleep(1)

    def execute_single_path(self):
        # todo: remove this sentence and modify the tt and ty
        vel = np.random.uniform(self.low_speed, self.high_speed, size=[3, ])

        # tt combination
        for mode in self.mask_t:
            self.tt(mode, vel, 10)

        # ty combination
        for mode in self.mask_t:
            yaw = np.random.uniform(low=0.5, high=8.0)
            self.ty(mode, vel, yaw, 10)

        # yy
        self.yy()

    def execute_sampling(self, num):

        # record the sample location
        cur_pos = self.cmd_client.getPosition()
        for i in range(num):
            # reset sequence idx for each sample
            self.seq_idx = 0

            # randomly start height
            height = np.random.uniform(0.3, 0.8)
            self.cmd_client.moveByVelocity(0.0, 0.0, -height, duration=1)
            time.sleep(1)

            # sample i-th data at this location
            self.execute_single_path()

            # reset the sample position
            self.cmd_client.moveToPosition(cur_pos.x_val, cur_pos.y_val, cur_pos.z_val, 0.5)

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


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.front_rgb = '1.front_rgb'
    args.bottom_rgb = '2.bottom_rgb'
    args.front_depth = '3.front_depth'
    args.bottom_depth = '4.bottom_depth'
    args.poses = '5.poses'

    sample_idx = 0
    root = join('/home/zhudelong/Videos/airsim', str(sample_idx))
    if exists(root):
        raise IOError('{} exist!'.format(root))

    os.mkdir(root)
    args.root = root
    ctrl = Controller(args)
    ctrl.execute_sampling(6)
    ctrl.cmd_client.reset()

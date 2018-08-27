import numpy as np

# dataset --- location ---- round ---- sequence ----- rgb/depth/pose      "TargetGamma": 2.0,
# fixed_anchor
'''
    anchors = np.array([[5914.829102, 79.554565, 538.634644, 0, 90],
                        [6695.843262, 752.442993, 522.522156, 0, 90],
                        [6124.726562, 5318.19873, 528.169495, 0, 90],
                        [7004.726562, 8368.203125, 578.171875, 30, -150],
                        [6188.813477, 7760.487793, 319.518463, 30, -120],
                        [-6111.811035, -6095.811035, 400.0, 30, -120],
                        [70.649345, 4719.407715, 486.207031, 30, -120],
                        [-484.956573, 3654.447998, 709.218933, 30, -120],
                        [-314.889832, -1760.927246, 502.103882, 30, -120],
                        [-813.773682, -1442.502686, 561.030823, 135, 0]])
'''
np.random.seed(666)


def neighbor_road():
    z_val = 650

    corner1 = np.array([-12490.0, 11110.0, 650.0])
    corner2 = np.array([12990.0, 11110.0, 650.0])
    corner3 = np.array([12990.0, -14380.0, 650.0])
    corner4 = np.array([-12490.0, -14380.0, 650.0])

    corner1_2 = [-12490.0, -1620.0, 650.0]
    corner3_4 = [12990.0, -1620.0, 650.0]

    corner2_3 = [250.000, 11110.0, 650.0]
    corner4_1 = [250.000, -14380.0, 650.0]

    way12 = get_way_points_x(corner1, corner2, z_val, 10, 'increase')
    way23 = get_way_points_y(corner2, corner3, z_val, 10, 'decrease')
    way34 = get_way_points_x(corner3, corner4, z_val, 10, 'decrease')
    way41 = get_way_points_y(corner4, corner1, z_val, 10, 'increase')

    corner1 = np.expand_dims(corner1, axis=0)
    corner2 = np.expand_dims(corner2, axis=0)
    corner3 = np.expand_dims(corner3, axis=0)
    corner4 = np.expand_dims(corner4, axis=0)
    out_road = np.concatenate([corner1, way12, corner2, way23, corner3, way34, corner4, way41, corner1], axis=0)
    # print out_road

    way12_34 = get_way_points_x(corner1_2, corner3_4, z_val, 10, 'increase')
    way23_41 = get_way_points_y(corner2_3, corner4_1, z_val, 10, 'decrease')

    corner1_2 = np.expand_dims(corner1_2, axis=0)
    corner3_4 = np.expand_dims(corner3_4, axis=0)

    corner2_3 = np.expand_dims(corner2_3, axis=0)
    corner4_1 = np.expand_dims(corner4_1, axis=0)
    cross_road = np.concatenate([corner1_2, way12_34, corner3_4, corner2_3, way23_41, corner4_1], axis=0)
    # print cross_road

    road_anchor = np.concatenate([out_road, cross_road], axis=0)
    print road_anchor
    return road_anchor


def get_way_points_x(corner1, corner2, z_val, anchor_num, order):
    way12_x = np.random.uniform(corner1[0], corner2[0], anchor_num)

    if order == 'increase':
        way12_x = np.sort(way12_x)
    elif order == 'decrease':
        way12_x = -np.sort(-way12_x)
    else:
        raise IOError('WRONG PARA')

    way12_y = np.ones(way12_x.shape) * corner1[1]
    way12_z = np.ones(way12_x.shape) * z_val
    way12 = np.stack([way12_x, way12_y, way12_z], axis=1)

    return way12


def get_way_points_y(corner1, corner2, z_val, anchor_num, order):
    way12_y = np.random.uniform(corner1[1], corner2[1], anchor_num)

    if order == 'increase':
        way12_y = np.sort(way12_y)
    elif order == 'decrease':
        way12_y = -np.sort(-way12_y)
    else:
        raise IOError('WRONG PARA')
    way12_x = np.ones(way12_y.shape) * corner1[0]
    way12_z = np.ones(way12_x.shape) * z_val
    way12 = np.stack([way12_x, way12_y, way12_z], axis=1)
    return way12


def get_anchors():
    # global anchors
    # map anchor coordinates
    anchors = np.array([[6460.178223, 668.619202, 700.778442, 20, -100],
                        [5790.064941, 188.838623, 739.244995, 40, -140],
                        [4783.101074, -688.609985, 719.003784, 30, -140],
                        [3701.682617, -1560.509155, 782.469177, 40, -130],
                        [2391.682617, -2430.509277, 772.469177, 30, 170],
                        [1291.682617, -2310.509277, 792.469177, -20, 160],
                        [-208.317383, -1790.509277, 792.469177, -30, 150],
                        [-648.317383, -1500.509277, 932.469177, -30, 150],
                        [-1278.317383, -1090.509277, 932.469177, -30, 140],
                        # intermidiate
                        [-1848.317383, -630.509277, 1012.469177, 90, -30],
                        # after cave
                        [-1848.317383, 1699.490723, 1292.469238, 50, -90],

                        [-1158.317383, 2789.490723, 1082.469177, -110, 50],
                        [-288.317383, 3879.490723, 1082.469177, -120, 60],
                        [-288.317383, 3879.490723, 1082.469177, -120, 60],
                        [441.682617, 4929.490723, 882.469177, -130, 30],
                        [1701.682617, 5489.490723, 722.469177, -170, 30],
                        [2931.682617, 6029.490723, 782.469177, 20, -160],
                        [4681.682617, 6559.490723, 782.469177, 50, -140],

                        [6821.682617, 8399.490234, 602.469177, 30, -150],
                        [6851.682617, 6729.490723, 602.469177, 50, -130],
                        [6231.682617, 5659.490723, 602.469177, 60, -100],
                        [6711.682617, 3859.490723, 602.469177, 100, -70],
                        [7121.682617, 1679.490723, 602.469177, 110, -140],
                        [7541.682617, -330.509277, 602.469177, -80, -140],
                        [8191.682617, -1840.509277, 602.469177, 110, -130],
                        [6871.682617, -3060.509277, 602.469177, 40, -150],
                        [5041.682617, -4360.509277, 602.469177, 50, 150],
                        [4951.682617, -2330.509277, 602.469177, 30, -130],
                        [6241.682617, -1070.509277, 602.469177, 40, -140],
                        ])

    # map start coordinates
    orig = np.array([5650, 730, 380]) / 100.0

    # Neighborhood
    # anchors = neighbor_road()
    # angel = np.random.uniform(-180, 180, anchors.shape)
    # anchors = np.concatenate([anchors, angel], axis=1)
    print anchors

    # map start coordinates
    # orig = np.array([-12120.0, 11320.0, 120]) / 100.0
    # get NED coordinates
    anchors[:, 0:3] = anchors[:, 0:3] / 100.0
    anchors[:, 0:3] = anchors[:, 0:3] - orig
    anchors[:, 2] = -anchors[:, 2]
    return anchors


if __name__ == '__main__':
    # print neighbor_road()
    print get_anchors()[57, :]

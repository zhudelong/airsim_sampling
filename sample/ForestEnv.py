import numpy as np

# dataset --- location ---- round ---- sequence ----- rgb/depth/pose
def get_anchors():
    global anchors
    # map anchor coordinates
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
    # map start coordinates
    orig = np.array([5650, 730, 380]) / 100.0
    # get NED coordinates
    anchors[:, 0:3] = anchors[:, 0:3] / 100.0
    anchors[:, 0:3] = anchors[:, 0:3] - orig
    anchors[:, 2] = -anchors[:, 2]
    return anchors

if __name__ == '__main__':
    print get_anchors()


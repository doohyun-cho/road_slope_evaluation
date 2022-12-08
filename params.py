import numpy as np

fx, fy = 2055.556149361639, 2055.556149361639
cx, cy = 939.6574698861468, 641.0721821943271
w_img = 1920
h_img = 1080
K = np.r_[[[fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]]]
K_inv = np.linalg.inv(K)

Z_cam = 2.115270572975561   # camera height(m)
v = 30.0    # vel, m/s
s = 1/30    # time interval, 30 fps
h = 0.01    # small value, for derivative calculation
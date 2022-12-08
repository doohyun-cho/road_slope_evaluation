import numpy as np
import cv2
from params import *

class Common:
    """
    Common utility class
    """

    def __init__(self):
        pass

    def rad2deg(self, x):
        return x * 180 / np.pi

    def deg2rad(self, x):
        return x * np.pi / 180

    def XYZ2uv_n(self, Xs, Ys, Zs, flag_bundle=False):
        uvs_n = np.ones((3, len(Xs)))
        uvs_n[0] = -Ys / Xs
        uvs_n[1] = (Zs-Z_cam) / Xs
        if flag_bundle:
            return uvs_n
        else:
            return uvs_n[0], uvs_n[1]

    def XYZ2xy(self, Xs, Ys, Zs, K=None, Z_offset=Z_cam, flag_bundle=False):
        if K is None:
            K = np.r_[[[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]]]
        YZXs = np.r_[[-Ys], 
                    [Z_offset-Zs], 
                    [Xs]]
        xys = K @ YZXs
        xys /= xys[2]
        if flag_bundle:
            return xys
        else:
            return xys[0], xys[1]
    
    # input xys dim: 2xn
    # output uvs dim: 2xn
    def xy2uv_n(self, xys):
        uvs_n = np.copy(xys)
        uvs_n[0] = (uvs_n[0]-cx)/fx
        uvs_n[1] = -(uvs_n[1]-cy)/fy
        return uvs_n

    # img to BEV coordinate conversion
    def xy2XY(self, xs, ys):
        us_n = (xs - cx) / fx
        vs_n = (-ys + cy) / fy
        Xs = Z_cam / (-vs_n)
        Ys = -Z_cam * us_n / (-vs_n)
        return Xs, Ys

    # img to BEV coordinate conversion
    def uv_n2XY(self, uvs_n):
        us_n = uvs_n[0]
        vs_n = uvs_n[1]
        Xs = Z_cam / (-vs_n)
        Ys = -Z_cam * us_n / (-vs_n)
        return Xs, Ys

    # img to BEV coordinate conversion
    def uv2XY(self, uvs):
        us_n = uvs[0] / fx
        vs_n = uvs[1] / fy
        Xs = Z_cam / (-vs_n)
        Ys = -Z_cam * us_n / (-vs_n)
        return Xs, Ys
    
    # input xys dim: 2xn
    # output uvs dim: 2xn
    def xy2uv(self, xys):
        uvs = np.copy(xys)
        uvs[0] = (uvs[0]-cx)
        uvs[1] = -(uvs[1]-cy)
        return uvs
    
    # input xys dim: 2xn
    # output uvs dim: 2xn
    def uv2uv_n(self, uvs):
        uvs_n = np.copy(uvs)
        uvs_n[0] /= fx
        uvs_n[1] /= fy
        return uvs_n

    
    # input xys dim: 2xn
    # output uvs dim: 2xn
    def uv_n2xy(self, uvs_n):
        xys = np.copy(uvs_n)
        xys[0] = xys[0]*fx + cx
        xys[1] = -xys[1]*fy + cy
        return xys

    # input xys dim: 2xn
    # output uv1s dim: 3xn
    def xy2uv1_ext(self, xys, H_ext):
        uvs = self.xy2uv(xys)
        uv1s = np.vstack((uvs, np.ones(uvs.shape[1])))
        uv1s_ext = H_ext @ uv1s
        uv1s_ext[0] /= uv1s_ext[2]
        uv1s_ext[1] /= uv1s_ext[2]
        uv1s_ext[2] = 1
        return uv1s_ext


    def cv2_undistort(self, srcPts, K, D):
        dstPts1 = cv2.undistortPoints(srcPts, K, D)
        dstPts2 = np.squeeze(dstPts1)
        dstPts3 = np.c_[dstPts2, np.ones(dstPts2.shape[0])]
        dstPts4 = (K@dstPts3.T).T
        return dstPts4[:,:2]
    
    """
    opencv2 Coord system
           
	    │ ∧ Z
		│/
     ───┼──> X
	   /│0
	  / │
        ▽Y
    """
    def R2rpy(self, R):
        pitch = np.arctan2(R[2,1], R[2,2])
        yaw = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
        roll = np.arctan2(R[1,0], R[0,0])
        return pitch, yaw, roll

    def rpy2R(self, r, p, y):
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        RX = np.r_[[[1, 0, 0]],
                   [[0, cp, -sp]],
                   [[0, sp, cp]]]
        RY = np.r_[[[cy, 0, -sy]],
                   [[0, 1, 0]],
                   [[sy, 0, cy]]]
        RZ = np.r_[[[cr, -sr, 0]],
                   [[sr, cr, 0]],
                   [[0, 0, 1]]]
        R = RX @ RY @ RZ
        return R

    def homography_from_ext_params(self, cam_data, flag_inv=False):
        r, p, y = cam_data['CAM_ROLL'], cam_data['CAM_PITCH'], cam_data['CAM_YAW']
        R = self.rpy2R(r, p, y)
        K = cam_data['INTRINSIC_MAT']
        K_inv = np.linalg.inv(K)
        
        H = K @ R @ K_inv
        if flag_inv:
            return np.linalg.inv(H)
        return H

    def homography_from_R(self, cam_data, R, flag_inv=False):
        K = cam_data['INTRINSIC_MAT']
        K_inv = np.linalg.inv(K)
        
        H = K @ R @ K_inv
        if flag_inv:
            return np.linalg.inv(H)
        return H

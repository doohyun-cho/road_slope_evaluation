#%%
import json
from multiprocessing import pool
import pickle

import os 
import io
from pathlib import Path 
from multiprocessing import *
from math import isclose
import time

import cv2
from PIL import Image
import PIL
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt
from tqdm import tqdm 
from common import Common

#%%
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)     # too much time consuming
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img = np.array(Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb()))[:,:,:3]

    return img

def get_coeffs(frame, frame_rsp, cam_params, poly):

    tZ = cam_params['CAM_TZ']
    # init poly prediction
    
    Xs_null_poly = poly.fit_transform(np.arange(4).reshape(-1,1))
    Ys_null = np.arange(4)
    gtY_l = LinearRegression().fit(Xs_null_poly, Ys_null)
    gtZ_l = LinearRegression().fit(Xs_null_poly, Ys_null)
    gtY_r = LinearRegression().fit(Xs_null_poly, Ys_null)
    gtZ_r = LinearRegression().fit(Xs_null_poly, Ys_null)
    rspY_l = LinearRegression().fit(Xs_null_poly, Ys_null)
    rspZ_l = LinearRegression().fit(Xs_null_poly, Ys_null)
    rspY_r = LinearRegression().fit(Xs_null_poly, Ys_null)
    rspZ_r = LinearRegression().fit(Xs_null_poly, Ys_null)

    gtY_l.coef_[:] = frame['left']['coeff_Y'][:]
    gtZ_l.coef_[:] = frame['left']['coeff_Z'][:]
    gtZ_l.coef_[0] += tZ
    gtY_r.coef_[:] = frame['right']['coeff_Y'][:]
    gtZ_r.coef_[:] = frame['right']['coeff_Z'][:]
    gtZ_r.coef_[0] += tZ
    
    rspY_l.coef_[:] = np.r_[frame_rsp['left']['coeff_Y'][:]]
    rspZ_l.coef_[:] = np.r_[frame_rsp['left']['coeff_Z'][:]]
    rspY_r.coef_[:] = np.r_[frame_rsp['right']['coeff_Y'][:]]
    rspZ_r.coef_[:] = np.r_[frame_rsp['right']['coeff_Z'][:]]

    return gtY_l, gtZ_l, gtY_r, gtZ_r, rspY_l, rspZ_l, rspY_r, rspZ_r


def calc_road_error(XYZs_l_gt_poly, XYZs_r_gt_poly, XYZs_l_rsp_poly, XYZs_r_rsp_poly, Xf_near=30):

    num_l_gt = XYZs_l_gt_poly.shape[1]
    num_r_gt = XYZs_r_gt_poly.shape[1]
    num_rsp = XYZs_l_rsp_poly.shape[1]
    X0_l_gt = XYZs_l_gt_poly[0,0]
    X0_r_gt = XYZs_r_gt_poly[0,0]
    X0_rsp = XYZs_l_rsp_poly[0,0]
    X0 = np.nan
    X0 = np.max([X0_l_gt, X0_r_gt, X0_rsp])
    X_cur = X0
    dX = XYZs_l_gt_poly[0,1] - XYZs_l_gt_poly[0,0]
    i_l_gt = np.int(X0 - X0_l_gt)
    i_r_gt = np.int(X0 - X0_r_gt)
    i_rsp = np.int(X0 - X0_rsp)

    errs_Y_near_l = []
    errs_Y_near_r = []
    errs_Y_far_l = []
    errs_Y_far_r = []
    errs_Z_near_l = []
    errs_Z_near_r = []
    errs_Z_far_l = []
    errs_Z_far_r = []
    Xf_near = 30

    while(i_rsp < num_rsp and i_l_gt < num_l_gt and i_r_gt < num_r_gt):
        if X_cur < Xf_near:
            errs_Y_near_l.append(np.abs(XYZs_l_gt_poly[1,i_l_gt] - XYZs_l_rsp_poly[1,i_rsp]))
            errs_Y_near_r.append(np.abs(XYZs_r_gt_poly[1,i_r_gt] - XYZs_r_rsp_poly[1,i_rsp]))
            errs_Z_near_l.append(np.abs(XYZs_l_gt_poly[2,i_l_gt] - XYZs_l_rsp_poly[2,i_rsp]))
            errs_Z_near_r.append(np.abs(XYZs_r_gt_poly[2,i_r_gt] - XYZs_r_rsp_poly[2,i_rsp]))
        else:
            errs_Y_far_l.append(np.abs(XYZs_l_gt_poly[1,i_l_gt] - XYZs_l_rsp_poly[1,i_rsp]))
            errs_Y_far_r.append(np.abs(XYZs_r_gt_poly[1,i_r_gt] - XYZs_r_rsp_poly[1,i_rsp]))
            errs_Z_far_l.append(np.abs(XYZs_l_gt_poly[2,i_l_gt] - XYZs_l_rsp_poly[2,i_rsp]))
            errs_Z_far_r.append(np.abs(XYZs_r_gt_poly[2,i_r_gt] - XYZs_r_rsp_poly[2,i_rsp]))
        i_rsp += 1
        i_l_gt += 1
        i_r_gt += 1
        X_cur += dX
    Xf = X_cur - dX
    
    errs_Y_near_l = np.r_[errs_Y_near_l]
    errs_Y_near_r = np.r_[errs_Y_near_r]
    errs_Z_near_l = np.r_[errs_Z_near_l]
    errs_Z_near_r = np.r_[errs_Z_near_r]
    errs_Y_far_l = np.r_[errs_Y_far_l]
    errs_Y_far_r = np.r_[errs_Y_far_r]
    errs_Z_far_l = np.r_[errs_Z_far_l]
    errs_Z_far_r = np.r_[errs_Z_far_r]

    mean_err_Y_near_l = None
    mean_err_Y_near_r = None
    mean_err_Z_near_l = None
    mean_err_Z_near_r = None
    if len(errs_Y_near_l) != 0:
        mean_err_Y_near_l = np.mean(errs_Y_near_l)
        mean_err_Y_near_r = np.mean(errs_Y_near_r)
        mean_err_Z_near_l = np.mean(errs_Z_near_l)
        mean_err_Z_near_r = np.mean(errs_Z_near_r)
    mean_err_Y_far_l = None
    mean_err_Y_far_r = None
    mean_err_Z_far_l = None
    mean_err_Z_far_r = None
    if len(errs_Y_far_l) != 0:
        mean_err_Y_far_l = np.mean(errs_Y_far_l)
        mean_err_Y_far_r = np.mean(errs_Y_far_r)
        mean_err_Z_far_l = np.mean(errs_Z_far_l)
        mean_err_Z_far_r = np.mean(errs_Z_far_r)

    mean_err_Y_l = None
    mean_err_Y_r = None
    mean_err_Z_l = None
    mean_err_Z_r = None
    if (len(errs_Y_near_l) + len(errs_Y_far_l)) != 0:
        mean_err_Y_l = np.mean(np.r_[errs_Y_near_l, errs_Y_far_l])
        mean_err_Y_r = np.mean(np.r_[errs_Y_near_r, errs_Y_far_r])
        mean_err_Z_l = np.mean(np.r_[errs_Z_near_l, errs_Z_far_l])
        mean_err_Z_r = np.mean(np.r_[errs_Z_near_r, errs_Z_far_r])

    return mean_err_Y_near_l, mean_err_Y_near_r, mean_err_Z_near_l, mean_err_Z_near_r, \
        mean_err_Y_far_l, mean_err_Y_far_r, mean_err_Z_far_l, mean_err_Z_far_r, \
        mean_err_Y_l, mean_err_Y_r, mean_err_Z_l, mean_err_Z_r, X0, Xf

def draw_result_on_video(fpath_input_video, fpath_output_video, fpath_scene, fpath_rsp, fpath_error):

    if not os.path.isfile(fpath_scene):
        print('Not exist -', fpath_scene)
        return
    if not os.path.isfile(fpath_rsp):
        print('Not exist -', fpath_rsp)
        return
    if os.path.isfile(fpath_output_video):
        print('already exists -', fpath_output_video)
        return 

    print('start -', fpath_input_video)

    with open(fpath_scene, 'rb') as f:
        scene = pickle.load(f)
    with open(fpath_rsp) as f:
        rsp = json.load(f)

    min_X_interval = 5.0
    vc = cv2.VideoCapture(fpath_input_video)
    w_fig, h_fig = 20, 12
    dpi = 100
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 30.0
    flag_init_vw = False

    # init img_out
    fig = plt.figure(constrained_layout=True, figsize=(w_fig, h_fig))
    axs = fig.subplot_mosaic([[0,1],
                            [2,3]],
                            gridspec_kw={'width_ratios':[7,1],
                                        'height_ratios':[5,1]})
    
    poly = PolynomialFeatures(degree=3, include_bias=True)
    font=cv2.FONT_HERSHEY_SIMPLEX

    pitch_deg = scene['cam_params']['CAM_PITCH']
    h_img = scene['cam_params']['PIXEL_HEIGHT']
    w_img = scene['cam_params']['PIXEL_WIDTH']
    fy = scene['cam_params']['CAM_FY']
    cy = scene['cam_params']['CAM_CY']
    tZ = scene['cam_params']['CAM_TZ']
    pixel_width_1m_ref = 1/tZ * (h_img - cy)
    pixel_width_1m = 1/tZ * (fy*np.tan(1.0 * np.pi / 180.0) + h_img - cy)
    pixel_pitch_change_rate = np.abs(1 - pixel_width_1m / pixel_width_1m_ref)
    rsp_pitch_ratio = 1 + pixel_pitch_change_rate * pitch_deg     # refer to oc_test_calc.xlsx

    mean_errs_Y_near_l = {}
    mean_errs_Y_near_r = {}
    mean_errs_Z_near_l = {}
    mean_errs_Z_near_r = {}
    mean_errs_Y_far_l = {}
    mean_errs_Y_far_r = {}
    mean_errs_Z_far_l = {}
    mean_errs_Z_far_r = {}
    mean_errs_Y_l = {}
    mean_errs_Y_r = {}
    mean_errs_Z_l = {}
    mean_errs_Z_r = {}

    for idx in range(np.min((len(scene['frames']), len(rsp)))):
        # if idx < 56:
        #     vc.grab()
        #     continue

        frame = scene['frames'][idx]
        frame_rsp = rsp[idx]
        
        if not (frame['flag_egolane_valid'] and frame_rsp['valid']):
            vc.grab()
        elif (frame['left']['Xf'] - frame['left']['X0'] < min_X_interval) or \
            (frame['right']['Xf'] - frame['right']['X0'] < min_X_interval):
            vc.grab()
        else:
            # with Timer('calc'):
            uvs_gt_l = frame['left']['uvs']
            uvs_gt_r = frame['right']['uvs']
            XYZs_gt_l = frame['left']['XYZs'][:,:uvs_gt_l.shape[1]]
            XYZs_gt_r = frame['right']['XYZs'][:,:uvs_gt_r.shape[1]]
            uvs_rsp_l = frame_rsp['left']['uvs']
            uvs_rsp_r = frame_rsp['right']['uvs']

            X_bottom = 7
            dX = 1.0
            Xs_rsp = np.arange(X_bottom, frame_rsp['Xf'], dX)   # 결과 비교는 정수 단위 (7, 8, 9, ...)으로 하고
            if len(Xs_rsp) <= 3:
                vc.grab()
                continue
            Xs_rsp_svnet = Xs_rsp / rsp_pitch_ratio             # rsp 계산은 이걸로 하고
            Xs_rsp_svnet_poly = poly.fit_transform(Xs_rsp_svnet.reshape(-1,1))

            gtY_l, gtZ_l, gtY_r, gtZ_r, rspY_l, rspZ_l, rspY_r, rspZ_r = \
                get_coeffs(frame, frame_rsp, scene['cam_params'], poly)

            Xs_l_gt = np.arange(np.ceil(frame['left']['X0']), frame['left']['Xf'], dX)
            Xs_r_gt = np.arange(np.ceil(frame['right']['X0']), frame['right']['Xf'], dX)
            Xs_l_gt_poly = poly.fit_transform(Xs_l_gt.reshape(-1,1))
            Xs_r_gt_poly = poly.fit_transform(Xs_r_gt.reshape(-1,1))
            Ys_l_gt = gtY_l.predict(Xs_l_gt_poly)
            Ys_r_gt = gtY_r.predict(Xs_r_gt_poly)
            Zs_l_gt = gtZ_l.predict(Xs_l_gt_poly)
            Zs_r_gt = gtZ_r.predict(Xs_r_gt_poly)

            XYZs_l_gt_poly = np.c_[[Xs_l_gt, Ys_l_gt, Zs_l_gt]]
            XYZs_r_gt_poly = np.c_[[Xs_r_gt, Ys_r_gt, Zs_r_gt]]

            # svnet3 돌릴 때 pitch 정보가 없기 때문에 d_lane 계산이 제대로 안 됐으므로
            # pitch 변화만큼 비율 보정해서 계산해 줘야 함
            Ys_l_rsp = rsp_pitch_ratio * rspY_l.predict(Xs_rsp_svnet_poly)
            Ys_r_rsp = rsp_pitch_ratio * rspY_r.predict(Xs_rsp_svnet_poly)
            Zs_l_rsp = rsp_pitch_ratio * rspZ_l.predict(Xs_rsp_svnet_poly)
            Zs_r_rsp = rsp_pitch_ratio * rspZ_r.predict(Xs_rsp_svnet_poly)
            
            XYZs_l_rsp_poly = np.c_[[Xs_rsp, Ys_l_rsp, Zs_l_rsp]]
            XYZs_r_rsp_poly = np.c_[[Xs_rsp, Ys_r_rsp, Zs_r_rsp]]

            mean_err_Y_near_l, mean_err_Y_near_r, mean_err_Z_near_l, mean_err_Z_near_r, \
                mean_err_Y_far_l, mean_err_Y_far_r, mean_err_Z_far_l, mean_err_Z_far_r, \
                mean_err_Y_l, mean_err_Y_r, mean_err_Z_l, mean_err_Z_r, X0, Xf = \
                calc_road_error(XYZs_l_gt_poly, XYZs_r_gt_poly, XYZs_l_rsp_poly, XYZs_r_rsp_poly)

            mean_errs_Y_near_l[idx] = mean_err_Y_near_l
            mean_errs_Y_near_r[idx] = mean_err_Y_near_r
            mean_errs_Z_near_l[idx] = mean_err_Z_near_l
            mean_errs_Z_near_r[idx] = mean_err_Z_near_r
            if mean_err_Y_far_l is not None:
                mean_errs_Y_far_l[idx] = mean_err_Y_far_l
                mean_errs_Y_far_r[idx] = mean_err_Y_far_r
                mean_errs_Z_far_l[idx] = mean_err_Z_far_l
                mean_errs_Z_far_r[idx] = mean_err_Z_far_r
            mean_errs_Y_l[idx] = mean_err_Y_l
            mean_errs_Y_r[idx] = mean_err_Y_r
            mean_errs_Z_l[idx] = mean_err_Z_l
            mean_errs_Z_r[idx] = mean_err_Z_r

            if not flag_init_vw:
                vw = cv2.VideoWriter(str(fpath_output_video), fourcc, fps, (w_fig*dpi, h_fig*dpi))
                flag_init_vw = True

            # with Timer('img read'):
            _, img = vc.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # with Timer('axs0'):
            ax = axs[0]
            ax.plot(uvs_gt_l[0], uvs_gt_l[1], 'b', label='gt')
            ax.plot(uvs_gt_r[0], uvs_gt_r[1], 'b')
            ax.plot(uvs_rsp_l[0], uvs_rsp_l[1], 'y', label='rsp')
            ax.plot(uvs_rsp_r[0], uvs_rsp_r[1], 'y')

            pl_gt = np.r_[uvs_gt_l[0][0]-200, uvs_gt_l[1][0]].astype(int)
            pr_gt = np.r_[uvs_gt_r[0][0], uvs_gt_r[1][0]].astype(int)
            pl_rsp = np.r_[uvs_gt_l[0][0]+30, h_img].astype(int)
            pr_rsp = np.r_[uvs_gt_r[0][0]-260, h_img].astype(int)
            cv2.putText(img, f'{XYZs_gt_l[1,0]:.3f}', pl_gt, font, 2, (0,0,255), 3)
            cv2.putText(img, f'{XYZs_gt_r[1,0]:.3f}', pr_gt, font, 2, (0,0,255), 3)

            cv2.putText(img, f'{Ys_l_rsp[0]:.3f}', pl_rsp, font, 2, (255,255,0), 3)
            cv2.putText(img, f'{Ys_r_rsp[0]:.3f}', pr_rsp, font, 2, (255,255,0), 3)
            ax.imshow(img)

            try:
                ax.text(30, 140, r'Error   $Y_{near}$      $Y$     $Z_{near}$      $Z$' + '\n'
                                f'  left: {mean_errs_Y_near_l[idx]:.3f}  {mean_errs_Y_l[idx]:.3f}  {mean_errs_Z_near_l[idx]:.3f}  {mean_errs_Z_l[idx]:.3f}\n' +
                                f'right: {mean_errs_Y_near_r[idx]:.3f}  {mean_errs_Y_r[idx]:.3f}  {mean_errs_Z_near_r[idx]:.3f}  {mean_errs_Z_r[idx]:.3f}\n' +
                                r'$X_0$~$X_f$: ' + f'{X0:.1f} ~ {Xf:.1f}',
                        fontsize=15, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
            except:
                ax.text(30, 140, r'Error   $Y_{near}$      $Y$     $Z_{near}$      $Z$' + '\n'
                                f'  left: {mean_errs_Y_near_l[idx]}  {mean_errs_Y_l[idx]}  {mean_errs_Z_near_l[idx]}  {mean_errs_Z_l[idx]}\n' +
                                f'right: {mean_errs_Y_near_r[idx]}  {mean_errs_Y_r[idx]}  {mean_errs_Z_near_r[idx]}  {mean_errs_Z_r[idx]}\n' +
                                r'$X_0$~$X_f$: ' + f'{X0} ~ {Xf}',
                        fontsize=15, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
                
            for u_l, v_l, u_r, v_r in list(zip(uvs_rsp_l[0], uvs_rsp_l[1], uvs_rsp_r[0], uvs_rsp_r[1])):
                ax.plot([u_l, u_r], [v_l, v_r], 'y')
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            ax.legend()

            # with Timer('axs1'):
            ax = axs[1]
            ax.plot(XYZs_gt_l[1], XYZs_gt_l[0], 'r.', label='gt left')
            ax.plot(XYZs_gt_r[1], XYZs_gt_r[0], 'b.', label='gt right')
            ax.plot(Ys_l_gt, Xs_l_gt, 'r:', linewidth=5, label='gt poly left')
            ax.plot(Ys_r_gt, Xs_r_gt, 'r-', linewidth=5, label='gt poly right')
            ax.plot(Ys_l_rsp, Xs_rsp, 'k:', linewidth=5, label='rsp')
            ax.plot(Ys_r_rsp, Xs_rsp, 'k-', linewidth=5)
            ax.invert_xaxis()
            ax.legend()
            ax.grid()

            # with Timer('axs2'):
            ax = axs[2]
            ax.plot(XYZs_gt_l[0], XYZs_gt_l[2] + tZ, 'r.', label='gt left')
            ax.plot(XYZs_gt_r[0], XYZs_gt_r[2] + tZ, 'b.', label='gt right')
            ax.plot(Xs_l_gt, Zs_l_gt, 'r:', linewidth=5, label='gt poly left')
            ax.plot(Xs_r_gt, Zs_r_gt, 'r-', linewidth=5, label='gt poly right')
            ax.plot(Xs_rsp, Zs_l_rsp, 'k:',  linewidth=5,label='rsp left')
            ax.plot(Xs_rsp, Zs_r_rsp, 'k-', linewidth=5, label='rsp right')
            ax.legend()
            ax.grid()
            plt.suptitle(f'frame {idx}')

            # plt.pause(0.1)

            # with Timer('fig2img'):
            img_fig = get_img_from_fig(fig)
            # with Timer('vw'):
            vw.write(img_fig)
            
            # with Timer('clear axs'):
            axs[0].cla()
            axs[1].cla()
            axs[2].cla()
            

    if not flag_init_vw:
        print(f'failed - {fpath_output_video}')
    else:
        errors = {}
        mean_errs_Y_near_l = np.r_[[x for x in mean_errs_Y_near_l if x is not None]]
        mean_errs_Y_near_r = np.r_[[x for x in mean_errs_Y_near_r if x is not None]]
        mean_errs_Z_near_l = np.r_[[x for x in mean_errs_Z_near_l if x is not None]]
        mean_errs_Z_near_r = np.r_[[x for x in mean_errs_Z_near_r if x is not None]]
        mean_errs_Y_far_l = np.r_[[x for x in mean_errs_Y_far_l if x is not None]]
        mean_errs_Y_far_r = np.r_[[x for x in mean_errs_Y_far_r if x is not None]]
        mean_errs_Z_far_l = np.r_[[x for x in mean_errs_Z_far_l if x is not None]]
        mean_errs_Z_far_r = np.r_[[x for x in mean_errs_Z_far_r if x is not None]]
        mean_errs_Y_l = np.r_[[x for x in mean_errs_Y_l if x is not None]]
        mean_errs_Y_r = np.r_[[x for x in mean_errs_Y_r if x is not None]]
        mean_errs_Z_l = np.r_[[x for x in mean_errs_Z_l if x is not None]]
        mean_errs_Z_r = np.r_[[x for x in mean_errs_Z_r if x is not None]]
        errors['mean_errs_Y_near_l'] = mean_errs_Y_near_l
        errors['mean_errs_Y_near_r'] = mean_errs_Y_near_r
        errors['mean_errs_Z_near_l'] = mean_errs_Z_near_l
        errors['mean_errs_Z_near_r'] = mean_errs_Z_near_r
        errors['mean_errs_Y_far_l'] = mean_errs_Y_far_l
        errors['mean_errs_Y_far_r'] = mean_errs_Y_far_r
        errors['mean_errs_Z_far_l'] = mean_errs_Z_far_l
        errors['mean_errs_Z_far_r'] = mean_errs_Z_far_r
        errors['mean_errs_Y_l'] = mean_errs_Y_l
        errors['mean_errs_Y_r'] = mean_errs_Y_r
        errors['mean_errs_Z_l'] = mean_errs_Z_l
        errors['mean_errs_Z_r'] = mean_errs_Z_r
        errors['mean_err_Y_l'] = np.mean(mean_errs_Y_l)
        errors['mean_err_Y_r'] = np.mean(mean_errs_Y_r)
        errors['mean_err_Z_l'] = np.mean(mean_errs_Z_l)
        errors['mean_err_Z_r'] = np.mean(mean_errs_Z_r)
        # locals().update(errors)       # <- Convert dictionary entries into variables
        
        with open(fpath_error, 'wb') as f:
            pickle.dump(errors, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'{len(mean_errs_Y_l)} frames - {fpath_output_video}')
        cv2.destroyAllWindows()
        vw.release()
    return



#%%

def main():
    root_training = {}
    root_validation = {}
    root_training['images'] = Path(r'D:\input\OpenLane\images\training')
    root_training['jsons'] = Path(r'D:\input\OpenLane\lane3d_1000\training')
    root_validation['images'] = Path(r'D:\input\OpenLane\images\validation')
    root_validation['jsons'] = Path(r'D:\input\OpenLane\lane3d_1000\validation')
    
    # target = '12161824480686739258_1813_380_1833_380_with_camera_labels'
    # fpath_video =     r'D:\input\OpenLane\images\training\segment-' + target + '.avi'
    # fpath_video_out = r'D:\input\OpenLane\images\training\segment-' + target + '_trash.avi'
    # fpath_scene =     r'D:\input\OpenLane\lane3d_1000\training\segment-' + target + '.pickle'
    # fpath_rsp =       r'D:\input\OpenLane\lane3d_1000\training\segment-' + target + '_rsp.json'
    # fpath_error =     r'D:\input\OpenLane\lane3d_1000\training\segment-' + target + '_error.pickle'
    # draw_result_on_video(fpath_video, fpath_video_out, fpath_scene, fpath_rsp, fpath_error)
    
    if True:
        pool = Pool(processes=11)
        print('training -')
        dirs_jsons = [d for d in next(os.walk(root_training['jsons']))[1]]
        args_list = [[str(root_training['images'] / dir_jsons) + '.avi', 
                     str(root_training['images'] / dir_jsons) + '_output.avi', 
                     str(root_training['jsons'] / dir_jsons) + '.pickle', 
                     str(root_training['jsons'] / dir_jsons) + '_rsp.json',
                     str(root_training['jsons'] / dir_jsons) + '_error.pickle'] 
                     for dir_jsons in dirs_jsons]
        r = pool.starmap(draw_result_on_video, tqdm(args_list, total=len(dirs_jsons)))        

        print('validation -')
        dirs_jsons = [d for d in next(os.walk(root_validation['jsons']))[1]]
        args_list = [[str(root_validation['images'] / dir_jsons) + '.avi', 
                     str(root_validation['images'] / dir_jsons) + '_output.avi', 
                     str(root_validation['jsons'] / dir_jsons) + '.pickle', 
                     str(root_validation['jsons'] / dir_jsons) + '_rsp.json',
                     str(root_validation['jsons'] / dir_jsons) + '_error.pickle'] 
                     for dir_jsons in dirs_jsons]
        r = pool.starmap(draw_result_on_video, tqdm(args_list, total=len(dirs_jsons)))

        pool.close()
        pool.join()
    return


#%%

if __name__ == '__main__':
    main()
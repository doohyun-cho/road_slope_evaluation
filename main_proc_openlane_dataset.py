#%%
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import cv2
import pickle
from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt
import json
from multiprocessing import *
from tqdm import tqdm
from common import Common

#%%

def jpgs2avi(root_dir, dir_jpgs, fps=30.0, verbose=False, flag_pass_if_exists=False):
    fvideo_name = str(dir_jpgs) + '.avi'
    fpath_video = root_dir / fvideo_name
    if flag_pass_if_exists:
        if os.path.exists(fpath_video):
            if verbose:
                print('exists -', fvideo_name)
            return 
        
    fpaths_img = list(Path(root_dir / dir_jpgs).glob('*.jpg'))

    # assume all images had same dimension
    img = cv2.imread(str(fpaths_img[0]))
    height, width, _ = img.shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')    
    vw = cv2.VideoWriter(str(fpath_video), fourcc, fps, (width, height))
    for fpath_img in fpaths_img:
        vw.write(cv2.imread(str(fpath_img)))
        
    cv2.destroyAllWindows()
    vw.release()
    if verbose:
        print('finished -', fvideo_name)
    
def laneinfo2camparam(fpath_camparam, lane_info, verbose=False, flag_rewrite=True):
    # assumption - carnival tire size = 235/60 R17
    # diameter: 714mm
    # half & pressed radius approximated as 350mm
    # tZ_half_tire = 0.350

    cam_params = {}
    K = np.r_[lane_info['intrinsic']]
    cam_params['CAM_FX'] = K[0,0]
    cam_params['CAM_FY'] = K[1,1]
    cam_params['CAM_CX'] = K[0,2]
    cam_params['CAM_CY'] = K[1,2]
    Tr = np.r_[lane_info['extrinsic']]
    R = Tr[:3,:3]
    roll_deg = np.arctan2(R[2,1], R[2,2]) * 180.0 / np.pi
    pitch_deg = -np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2)) * 180.0 / np.pi
    yaw_deg = np.arctan2(R[1,0], R[0,0]) * 180.0 / np.pi
    cam_params['CAM_YAW'] = roll_deg
    cam_params['CAM_PITCH'] = pitch_deg
    cam_params['CAM_ROLL'] = yaw_deg
    cam_params['CAM_TX'] = 0        # Tr[0,3]
    cam_params['CAM_TY'] = 0        # Tr[1,3]
    # cam_params['CAM_TZ'] = Tr[2,3] + tZ_half_tire
    cam_params['CAM_TZ'] = Tr[2,3]
    cam_params['CAM_K1'] = 0.032316008498
    cam_params['CAM_K2'] = -0.321412482553
    cam_params['CAM_K3'] = 0.0
    cam_params['CAM_K4'] = 0.0
    cam_params['CAM_P1'] = 0.000793258395371
    cam_params['CAM_P2'] = -0.000625749354133
    cam_params['CAM_DISTORTION'] = 'RECTILINEAR'
    cam_params['PIXEL_WIDTH'] = 1920
    cam_params['PIXEL_HEIGHT'] = 1280

    if os.path.exists(fpath_camparam) and not flag_rewrite:
        if verbose:
            print('exists -' + str(fpath_camparam))
    else:    
        with open(fpath_camparam, 'w') as f:
            # put 0 for svnet3 execution
            cam_params_ = cam_params.copy()
            cam_params_['CAM_YAW'] = 0
            cam_params_['CAM_PITCH'] = 0
            cam_params_['CAM_ROLL'] = 0
            for key in cam_params_.keys():
                f.write(f'{key} = {cam_params_[key]}\n')
        if verbose:
            print('finished -' + str(fpath_camparam))
        
    return cam_params

# delete line if it's curbside
def delete_curbside(lane_lines):
    for lane_line in lane_lines:
        if lane_line['category'] == 20 or lane_line['category'] == 21:
            lane_lines.remove(lane_line)
    return lane_lines

# connections list has a index pair 
# if one of the line segment start position is near the end position of the other's
# or one of the line segment end position is near the start position of the other's
def find_line_connections(lane_lines, thr_x=3.0, thr_yz=0.8):
    connections = []
    for i0 in range(len(lane_lines)):
        for i1 in range(i0 + 1, len(lane_lines)):
            xyz_i0_start = np.r_[lane_lines[i0]['xyz']][:,0]
            xyz_i1_start = np.r_[lane_lines[i1]['xyz']][:,0]
            
            # sometimes the last element is [0,0,0]
            i_end = -1
            while np.r_[lane_lines[i0]['xyz']][0,i_end] == 0:
                i_end -= 1
            xyz_i0_end = np.r_[lane_lines[i0]['xyz']][:,i_end]
            
            i_end = -1
            while np.r_[lane_lines[i1]['xyz']][0,i_end] == 0:
                i_end -= 1
            xyz_i1_end = np.r_[lane_lines[i1]['xyz']][:,i_end]
            
            # compare start and end point pair
            # more threshold on X dist, less threshold on YZ dist
            xyz_01 = np.abs(xyz_i0_end - xyz_i1_start)
            if xyz_01[0] < thr_x and np.linalg.norm(xyz_01[1:]) < thr_yz:
                connections += [[i0, i1]]
                
            xyz_10 = np.abs(xyz_i0_start - xyz_i1_end)
            if xyz_10[0] < thr_x and np.linalg.norm(xyz_10[1:]) < thr_yz:
                connections += [[i1, i0]]
    
    for conn0 in reversed(connections):
        for conn1 in reversed(connections[:-1]):

            if conn0[-1] == conn1[0]:
                conn0 += conn1[1:]
                if conn1 in connections:
                    connections.remove(conn1)
                continue

            if conn1[-1] == conn0[0]:
                conn1 += conn0[1:]
                if conn0 in connections:
                    connections.remove(conn0)
                continue
            
    return connections

def uv_equals_XYZ2xy(uvs, XYZs, K, Z_offset=0, thr_pixel=1.0):
    cm = Common()
    xs, ys = cm.XYZ2xy(XYZs[0], XYZs[1], XYZs[2], K=K, Z_offset=Z_offset, flag_bundle=False)
    return mean_squared_error(xs, uvs[0], squared=False) < thr_pixel and mean_squared_error(ys, uvs[1], squared=False) < thr_pixel

def XYZ_results(XYZs, X_bottom=8.0, dX=1.0):
    Xs, Ys, Zs = XYZs
    X0, Xf = Xs[0], Xs[-1]

    poly = PolynomialFeatures(degree=3, include_bias=True)
    Xs_poly = poly.fit_transform(Xs.reshape(-1,1))
    linreg_Y = LinearRegression().fit(Xs_poly, Ys)
    linreg_Z = LinearRegression().fit(Xs_poly, Zs)
    Ys_poly = linreg_Y.predict(Xs_poly)
    Zs_poly = linreg_Z.predict(Xs_poly)
    coeff_Y = np.r_[linreg_Y.intercept_, linreg_Y.coef_[1:]]
    coeff_Z = np.r_[linreg_Z.intercept_, linreg_Z.coef_[1:]]
    
    X0_poly = poly.transform(np.r_[[[X0]]])
    X1_poly = poly.transform(np.r_[[[X0 + dX]]])
    Y0_pred = linreg_Y.predict(X0_poly)
    Y1_pred = linreg_Y.predict(X1_poly)
    dYdX = (Y1_pred - Y0_pred) / dX
    Y_bottom_pred = Y0_pred + dYdX * (X_bottom - X0)
    std_Y = np.std((Ys_poly - Ys))
    std_Z = np.std((Zs_poly - Zs))
    return X0, Xf, coeff_Y, coeff_Z, Y_bottom_pred, std_Y, std_Z

def lane_jsons2refined_dictpickle(
    root_json, 
    flag_camparam=True, 
    verbose=False,
    max_X0=20.0, 
    max_abs_Y_bottom=4.0,
    min_lane_bottom_dist=2.0
    ):
    
    rootroot_json, scene_name = os.path.split(root_json)
    fpath_camparam = Path(rootroot_json) / (scene_name + '_cam_param.ini')
    fpath_scene = Path(rootroot_json) / (scene_name + '.pickle')

    fjson_names = [d for d in next(os.walk(root_json))][2]
    if len(fjson_names) == 0:
        print('no appropriate json files in', root_json)
        return

    # use only first frame json file for cam_param.ini
    with open(root_json / fjson_names[0]) as f:
        lane_info = json.load(f)
    
    if flag_camparam:
        cam_params = laneinfo2camparam(fpath_camparam, lane_info, verbose, flag_rewrite=True)
    K = np.r_[[[cam_params['CAM_FX'], 0, cam_params['CAM_CX']],
               [0, cam_params['CAM_FY'], cam_params['CAM_CY']],
               [0, 0, 1]]]
    Z_offset = cam_params['CAM_TZ']

    scene = {}
    scene['at_least_1frame_valid'] = False
    scene['frames'] = {}
    scene['name'] = scene_name
    if flag_camparam:
        scene['cam_params'] = cam_params

    for i, fjson_name in enumerate(fjson_names):
        with open(root_json / fjson_name) as f:
            lane_info = json.load(f)
        
        frame = {}
        frame['name'] = Path(fjson_name).stem
        lane_lines = lane_info['lane_lines']
        lane_lines = delete_curbside(lane_lines)
        connections = find_line_connections(lane_lines)
        # first, get lines in the connections list
        # and then add the rest
        lines_info = []
        line_info_keys = ['uvs', 'XYZs', 'X0', 'Xf', 'coeff_Y', 'coeff_Z', 'Y_bottom', 'std_Y', 'std_Z', 'flag_uv_equals_XYZ2xy']
        line_idxs_left = list(range(len(lane_lines)))
        for line_idxs in connections:
            XYZs_bundle = np.zeros((3,0))
            uvs_bundle = np.zeros((2,0))
            for idx in line_idxs:
                visibility = np.r_[lane_lines[idx]['visibility']].astype(bool)
                XYZs = np.r_[lane_lines[idx]['xyz']]
                uvs = np.r_[lane_lines[idx]['uv']]
                num_XYZs = 0
                while(num_XYZs < len(visibility) and visibility[num_XYZs] == True):
                    num_XYZs += 1
                num_uvs = uvs.shape[1]
                num_dup = np.min((num_XYZs, num_uvs))
                XYZs = XYZs[:,:num_dup]
                uvs = uvs[:,:num_dup]

                XYZs_bundle = np.hstack((XYZs_bundle, XYZs))
                uvs_bundle = np.hstack((uvs_bundle, uvs))
                
                if idx in line_idxs_left:
                    line_idxs_left.remove(idx)
            
            X0, Xf, coeff_Y, coeff_Z, Y_bottom, std_Y, std_Z = \
                XYZ_results(XYZs_bundle)
            
            flag_uv_equals_XYZ2xy = uv_equals_XYZ2xy(uvs, XYZs, K)
            values = [uvs_bundle, XYZs_bundle, X0, Xf, coeff_Y, coeff_Z, Y_bottom, std_Y, std_Z, flag_uv_equals_XYZ2xy]
            line_info = dict(zip(line_info_keys, values))
            lines_info.append(line_info)
        
        for idx in line_idxs_left:
            visibility = np.r_[lane_lines[idx]['visibility']].astype(bool)
            XYZs = np.r_[lane_lines[idx]['xyz']]
            uvs = np.r_[lane_lines[idx]['uv']]
            num_XYZs = 0
            while(num_XYZs < len(visibility) and visibility[num_XYZs] == True):
                num_XYZs += 1
            num_uvs = uvs.shape[1]
            num_dup = np.min((num_XYZs, num_uvs))
            XYZs = XYZs[:,:num_dup]
            uvs = uvs[:,:num_dup]
            
            X0, Xf, coeff_Y, coeff_Z, Y_bottom, std_Y, std_Z = \
                XYZ_results(XYZs)

            flag_uv_equals_XYZ2xy = uv_equals_XYZ2xy(uvs, XYZs, K)
            values = [uvs, XYZs, X0, Xf, coeff_Y, coeff_Z, Y_bottom, std_Y, std_Z, flag_uv_equals_XYZ2xy]
            line_info = dict(zip(line_info_keys, values))
            lines_info.append(line_info)
        
        idx_left = -1        
        idx_right = -1
        left_min_dist = np.inf
        right_min_dist = np.inf
        for idx, line_info in enumerate(lines_info):
            X0 = line_info['X0']
            Y_bottom = line_info['Y_bottom']
            if X0 > max_X0:
                continue
            if np.abs(Y_bottom) > max_abs_Y_bottom:
                continue

            if Y_bottom > 0 and np.abs(Y_bottom) < left_min_dist:
                idx_left = idx
                left_min_dist = np.abs(Y_bottom)
            if Y_bottom <= 0 and np.abs(Y_bottom) < right_min_dist:
                idx_right = idx
                right_min_dist = np.abs(Y_bottom)
        
        if idx_left != -1:
            frame['left'] = lines_info[idx_left]
        if idx_right != -1:
            frame['right'] = lines_info[idx_right]

        # filter results if lane info is not appropriate
        frame['flag_egolane_valid'] = True
        if idx_left is -1 or idx_right is -1:
            frame['flag_egolane_valid'] = False
        if (left_min_dist + right_min_dist) < min_lane_bottom_dist:
            frame['flag_egolane_valid'] = False      

        if frame['flag_egolane_valid']:
            scene['at_least_1frame_valid'] = True
            
        scene['frames'][i] = frame
    Tr = np.r_[lane_info['extrinsic']]
    R = Tr[:3,:3]
    scene['Tr'] = Tr
    scene['R'] = R

    with open(fpath_scene, 'wb') as f:
        pickle.dump(scene, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    if verbose:
        print('finished -', fpath_scene)
    return scene

def write_svnet_batch(root_jsons, root_images, flag_training):
    if flag_training:              
        fpath_batch = root_jsons / 'svnet3_openlane_training.bat'
    else:
        fpath_batch = root_jsons / 'svnet3_openlane_validation.bat'
    str_exe = r'D:\dev\Stradvision\svnet3\build\references\sdk_v4\Release\svnet3_sample_case_1.exe'
    with open(fpath_batch, 'w') as f_bat:
        fpaths_scene = list(Path(root_jsons).glob('*.pickle'))
        for i, fpath_scene in tqdm(enumerate(fpaths_scene)):
            with open(fpath_scene, 'rb') as f:
                scene = pickle.load(f)
                if scene['at_least_1frame_valid']:
                    scene_name = fpath_scene.stem
                    str_video = str(root_images / str(scene_name + '.avi'))
                    str_camini = str(root_jsons / str(scene_name + '_cam_param.ini'))
                    str_can = '-input_can_data ' + str(root_jsons / str(scene_name + '_can.csv'))
                    f_bat.write(f'{str_exe} {str_video} {str_camini} {str_can}\n')

#%%    

root_images_training = Path(r'D:\input\OpenLane\images\training')
root_images_validation = Path(r'D:\input\OpenLane\images\validation')
root_jsons_training = Path(r'D:\input\OpenLane\lane3d_1000\training')
root_jsons_validation = Path(r'D:\input\OpenLane\lane3d_1000\validation')

def main():
    pool = Pool(processes=12)
    
    if False:
        print('training -')
        dirs_jpgs = [d for d in next(os.walk(root_images_training))[1]]
        arg_list = [[root_images_training, dir_jpgs] for dir_jpgs in dirs_jpgs]
        r = pool.starmap(jpgs2avi, tqdm(arg_list, total=len(dirs_jpgs)))
        # r = list(tqdm(pool.starmap(jpgs2avi, 
        #         [[root_images_training, dir_jpgs] for dir_jpgs in dirs_jpgs]), total=len(dirs_jpgs)))
        # for dir_jpgs in dirs_jpgs:
        #     pool.apply_async(jpgs2avi, args=[root_images_training, dir_jpgs])
        
        print('validation -')
        dirs_jpgs = [d for d in next(os.walk(root_images_validation))[1]]
        arg_list = [[root_images_validation, dir_jpgs] for dir_jpgs in dirs_jpgs]
        r = pool.starmap(jpgs2avi, tqdm(arg_list, total=len(dirs_jpgs)))
        # r = list(tqdm(pool.imap(jpgs2avi, 
        #         [[root_images_validation, dir_jpgs] for dir_jpgs in dirs_jpgs]), total=len(dirs_jpgs)))
        # for dir_jpgs in dirs_jpgs:
        #     pool.apply_async(jpgs2avi, args=[root_images_validation, dir_jpgs])

    if True:
        print('training -')
        dirs_jsons = [d for d in next(os.walk(root_jsons_training))[1]]
        arg_list = [root_jsons_training / dir_jsons for dir_jsons in dirs_jsons]
        # for args in arg_list:
            # if 'segment-12161824480686739258_1813_380_1833_380_with_camera_labels' == args.stem:
            # lane_jsons2refined_dictpickle(args)
        r = pool.imap(lane_jsons2refined_dictpickle, tqdm(arg_list, total=len(dirs_jsons)))

        print('validation -')
        dirs_jsons = [d for d in next(os.walk(root_jsons_validation))[1]]
        arg_list = [root_jsons_validation / dir_jsons for dir_jsons in dirs_jsons]
        r = pool.imap(lane_jsons2refined_dictpickle, tqdm(arg_list, total=len(dirs_jsons)))

    # read scene pickle files - around 10 secs for 1000 files
    if False:
        write_svnet_batch(root_jsons_training, root_images_training, flag_training=True)
        write_svnet_batch(root_jsons_validation, root_images_validation, flag_training=False)

    pool.close()
    pool.join()

    # for a single test
    if False:
        flag_draw_img = True
        if flag_draw_img:
            plt.figure(figsize=(12, 7))
        dirs_jsons = [d for d in next(os.walk(root_jsons_training))[1]]
        
        cnt = 0
        for dir_jsons in dirs_jsons:
            if cnt <= 3:
                cnt += 1
                continue
            # # if dir_jsons != 'segment-8722413665055769182_2840_000_2860_000_with_camera_labels':
            # if dir_jsons != 'segment-8745106945249251942_1207_000_1227_000_with_camera_labels':
            #     continue
            scene = lane_jsons2refined_dictpickle(root_jsons_training / dir_jsons)
            if flag_draw_img:
                font=cv2.FONT_HERSHEY_SIMPLEX
                
                for idx in range(len(scene['frames'])):
                    frame = scene['frames'][idx]
                    img = cv2.imread(str(root_images_training / dir_jsons / frame['name']) + '.jpg')
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    if 'left' in frame:
                        uvs = frame['left']['uvs']
                        p0 = uvs[:,0].astype(int)
                        cv2.putText(img, 'left', p0, font, 2, (255,255,0), 3)
                        for i in range(1, uvs.shape[1]):
                            p1 = uvs[:,i].astype(int)
                            cv2.line(img, p0, p1, (255,255,0), 5)
                            p0 = p1

                    if 'right' in frame:
                        uvs = frame['right']['uvs']
                        p0 = uvs[:,0].astype(int)
                        cv2.putText(img, 'right', p0, font, 2, (0,0,255), 3)
                        for i in range(1, uvs.shape[1]):
                            p1 = uvs[:,i].astype(int)
                            cv2.line(img, p0, p1, (0,0,255), 5)
                            p0 = p1

                    plt.clf()
                    plt.imshow(img)
                    plt.pause(0.0001)

            break

    # D:\dev\Stradvision\svnet3\build\references\sdk_v4\Release\svnet3_sample_case_1.exe "D:\input\OpenLane\images\training\segment-15832924468527961_1564_160_1584_160_with_camera_labels.avi" "D:\input\OpenLane\lane3d_1000\training\segment-15832924468527961_1564_160_1584_160_with_camera_labels_cam_param.ini"
if __name__ == '__main__':
    main()
# %%

import open3d as o3d 
from plyfile import PlyData, PlyElement
import numpy as np 
import os 
from ply import write_ply
import argparse

parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')

parser.add_argument('--ROOT_PATH', default='datasets/ParisLille-3D')
parser.add_argument('--n_train', default=5, type=int)
parser.add_argument('--n_test', default=5, type=int)
args = parser.parse_args()

def read_ParisLille3D_format(filename):
    """convert from a ply file. include the label and the object number"""
    #---read the ply file--------
    plydata = PlyData.read(filename)
    xyz = np.stack([plydata['vertex'][n] for n in['x', 'y', 'z']], axis=1)
    try:
        labels = plydata['vertex']['class']
        return xyz, labels
    except ValueError:
        return xyz

path = args.ROOT_PATH + '/data/' 
intermediate_folder = "pointclouds/"
names = ["Lille1_1.ply","Lille1_2.ply","Lille2.ply","Paris.ply","ajaccio_2.ply","ajaccio_57.ply", "dijon_9.ply"]
for pc in names:
    print(pc)
    if not pc in ["ajaccio_2.ply","ajaccio_57.ply", "dijon_9.ply"]:
        N_SUBSAMPLE = args.n_train
        _, labels = read_ParisLille3D_format(path+intermediate_folder+pc)
        pcd = o3d.io.read_point_cloud(path+intermediate_folder+pc)
        pcd.colors = o3d.utility.Vector3dVector(
            np.ones(shape=(labels.shape[0], 3)) * labels[:,None])
        STEP = len(np.asarray(labels))//N_SUBSAMPLE
        del labels
    else:
        N_SUBSAMPLE = args.n_test
        pcd = o3d.io.read_point_cloud(path+intermediate_folder+pc)
        STEP = len(np.asarray(pcd.points))//N_SUBSAMPLE


    if not os.path.exists(path + pc[:-4] + '/'):
        os.makedirs(path + pc[:-4])

    for k in range(N_SUBSAMPLE):
        small_pcd = o3d.geometry.PointCloud()
        if k < N_SUBSAMPLE-1:
            small_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[k*STEP: (k+1)*STEP])
            try:
                small_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[k*STEP: (k+1)*STEP])
            except:
                pass
        else:
            small_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[k*STEP: ])
            try:
                small_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[k*STEP: ])
            except:
                pass
        if not pc in ["ajaccio_2.ply","ajaccio_57.ply", "dijon_9.ply"]:
            classes = np.asarray(small_pcd.colors)[:,0].astype(np.uint8)
            write_ply(path + pc[:-4] + '/' + pc[:-4] + '_' + str(k) + '.ply', np.hstack((np.asarray(small_pcd.points)[classes != 0], classes.reshape(-1,1)[classes != 0])), ['x', 'y', 'z', 'class'])
        else:
            write_ply(path + pc[:-4] + '/' + pc[:-4] + '_' + str(k) + '.ply', np.asarray(small_pcd.points), ['x', 'y', 'z'])

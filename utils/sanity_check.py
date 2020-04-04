import numpy as np 
from plyfile import PlyData, PlyElement
from ply import write_ply
import os, glob
def read_ParisLille3D_format(filename, label=True):
    """convert from a ply file. include the label and the object number"""
    #---read the ply file--------
    plydata = PlyData.read(filename)
    xyz = np.stack([plydata['vertex'][n] for n in['x', 'y', 'z']], axis=1)
    if label:
        labels = plydata['vertex']['class']
        return xyz, labels
    else:
        return xyz
PATH = "datasets/ParisLille-3D/labels/"
NAMES = ['Paris']
PATH_CLOUD = "datasets/ParisLille-3D/data/pointclouds/"
root = 'datasets/ParisLille-3D'+'/'
test_folders = ['Paris/']
# for area in test_folders:
#     data_folder = root + "data/"               + area
#     labels_folder =  root + "labels/"          + area
#     files = glob.glob(data_folder+"*.ply")  
#     for file in files:
#         file_name = os.path.splitext(os.path.basename(file))[0]
#         file_name_short = '_'.join(file_name.split('_')[:3])
#         data_file  = data_folder + file_name + ".ply"
#         label_file = labels_folder + file_name_short + ".labels"
#         xyz_up = read_ParisLille3D_format(data_file,False)
#         labels = np.loadtxt(label_file)
#         write_ply(data_folder +file_name+'_labeled.ply', np.hstack((xyz_up, labels.reshape(-1,1))), ['x', 'y', 'z', 'class'])
for n in NAMES:
    print(n)
    local_path = PATH + n + '/' + n + '.labels'
    labels = np.loadtxt(local_path)
    print(np.unique(labels))
    plydata = PlyData.read(PATH_CLOUD+n+'.ply')
    xyz = np.stack([plydata['vertex'][n] for n in['x', 'y', 'z']], axis=1)
    classes = plydata['vertex']['class']
    write_ply(PATH_CLOUD+n+'_labeled.ply', np.hstack((xyz[classes != 0], labels.reshape(-1,1))), ['x', 'y', 'z', 'class'])

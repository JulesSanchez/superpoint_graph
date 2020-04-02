import numpy as np 
from plyfile import PlyData, PlyElement
from ply import write_ply

PATH = "datasets/ParisLille-3D/labels/"
NAMES = ['ajaccio_2','ajaccio_57','dijon_9']
PATH_CLOUD = "datasets/ParisLille-3D/data/pointclouds/"

for n in NAMES:
    local_path = PATH + n + '/' + n + '.labels'
    labels = np.loadtxt(local_path)
    plydata = PlyData.read(PATH_CLOUD+n+'.ply')
    xyz = np.stack([plydata['vertex'][n] for n in['x', 'y', 'z']], axis=1)
    write_ply(PATH_CLOUD+n+'_labeled.ply', np.hstack((xyz, labels.reshape(-1,1))), ['x', 'y', 'z', 'class'])

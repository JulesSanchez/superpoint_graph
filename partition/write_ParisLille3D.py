#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    
call this function once the partition and inference was made to upsample
the prediction to the original point clouds
"""
import os.path
import glob
import numpy as np
import argparse
from provider import *
parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
parser.add_argument('--PARISLILLE3D_PATH', default='datasets/ParisLille-3D')
parser.add_argument('--odir', default='./results/ParisLille-3D', help='Directory to store results')
parser.add_argument('--ver_batch', default=5000000, type=int, help='Batch size for reading large files')
args = parser.parse_args()
#---path to data---------------------------------------------------------------
#root of the data directory
root = args.PARISLILLE3D_PATH+'/'
res_folder  = './' + args.odir + '/'
#list of subfolders to be processed
test_folders = ['ajaccio_2/','ajaccio_57/','dijon_9/']
res_file = h5py.File(res_folder + 'predictions_test' + '.h5', 'r')   
for area in test_folders:
#------------------------------------------------------------------------------
    print("=================\n   " + area + "\n=================")
    data_folder = root + "data/"               + area
    fea_folder  = root + "features/"           + area
    spg_folder  = root + "superpoint_graphs/"           + area
    labels_folder =  root + "labels/"          + area
    if not os.path.isdir(data_folder):
        raise ValueError("%s do not exists" % data_folder)
    if not os.path.isdir(fea_folder):
        raise ValueError("%s do not exists" % fea_folder)
    if not os.path.isdir(res_folder):
        raise ValueError("%s do not exists" % res_folder)  
    if not os.path.isdir(root + "labels/"):
        os.mkdir(root + "labels/")   
    if not os.path.isdir(labels_folder):
        os.mkdir(labels_folder) 
    local_res_file = res_file.get(area[:-1])  
        
    files = glob.glob(data_folder+"*.ply")    
    if (len(files) == 0):
        raise ValueError('%s is empty' % data_folder)
    n_files = len(files)
    i_file = 0
    labels_name = []
    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name_short = '_'.join(file_name.split('_')[:3])
        data_file  = data_folder + file_name + ".ply"
        fea_file   = fea_folder  + file_name_short + '.h5'
        spg_file   = spg_folder  + file_name_short + '.h5' 
        label_file = labels_folder + file_name_short + ".labels"
        labels_name.append(label_file)
        i_file = i_file + 1
        print(str(i_file) + " / " + str(n_files) + "---> "+file_name_short)
        print("    reading the subsampled file...")
        geof, xyz, rgb, graph_nn, l = read_features(fea_file)
        graph_sp, components, in_component = read_spg(spg_file)
        n_ver = xyz.shape[0]
        del geof, rgb, graph_nn, l, graph_sp
        labels_red = np.array(local_res_file.get(file_name_short))
        print("    upsampling...")
        labels_full = reduced_labels2full(labels_red, components, n_ver)
        xyz_up = read_ParisLille3D_format(data_file,False)
        labels_ups = interpolate_labels(xyz_up, xyz, labels_full, args.ver_batch)
        np.savetxt(label_file, labels_ups+1, delimiter=' ', fmt='%d')   # X is an array
    print("     Concatenate into output file...")
    labels_name.sort()
    label_file = labels_folder + area + ".labels"
    with open(label_file, 'w') as outfile:
        for fname in labels_name:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

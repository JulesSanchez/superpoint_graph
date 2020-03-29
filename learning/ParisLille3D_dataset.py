    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:16:14 2018

@author: landrieuloic
""""""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    Template file for processing custome datasets
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import spg


def get_datasets(args, test_seed_offset=0):
    """build training and testing set"""
    train_names  =["Lille1_1/","Lille1_2/","Paris/"]
    val_names = ["Lille2/"]
    test_names = ["ajaccio_2/","ajaccio_57/", "dijon_9/"]
    
    # Load superpoints graphs
    testlist, trainlist, validlist = [], [], []
    for n in train_names:
        for k in os.listdir(args.PARISLILLE3D_PATH + '/superpoint_graphs/' + n ):
            trainlist.append(spg.spg_reader(args, args.PARISLILLE3D_PATH + '/superpoint_graphs/' + n + os.path.splitext(k)[0]+ '.h5', True))
    for n in test_names:
        for k in os.listdir(args.PARISLILLE3D_PATH + '/superpoint_graphs/' + n ):
            testlist.append(spg.spg_reader(args, args.PARISLILLE3D_PATH + '/superpoint_graphs/' + n + os.path.splitext(k)[0]+'.h5', True))
    for n in val_names:
        for k in os.listdir(args.PARISLILLE3D_PATH + '/superpoint_graphs/' + n ):
            validlist.append(spg.spg_reader(args, args.PARISLILLE3D_PATH + '/superpoint_graphs/' + n + os.path.splitext(k)[0]+'.h5', True))

    # Normalize edge features
    if args.spg_attribs01:
       trainlist, testlist, validlist, scaler = spg.scaler01(trainlist, testlist, validlist=validlist)

    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.PARISLILLE3D_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.PARISLILLE3D_PATH, test_seed_offset=test_seed_offset)) ,\
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in validlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.PARISLILLE3D_PATH, test_seed_offset=test_seed_offset)),\
            scaler

def get_info(args):
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1

    return {
        'node_feats': 11 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'classes': 10, 
        'inv_class_map': {0 :"unclassified",
                        1 :"ground",
                        2 :"building",
                        3 :"pole - road sign - traffic light",
                        4 :"bollard - small pole",
                        5 :"trash can",
                        6 :"barrier",
                        7 :"pedestrian",
                        8 :"car",
                        9 :"natural - vegetation"}
    }

def preprocess_pointclouds(SEMA3D_PATH):
    """ Preprocesses data by splitting them by components and normalizing."""

    for n in ["ajaccio_2/","Lille1_1/","Lille1_2/","Lille2/","Paris/","ajaccio_57/", "dijon_9/"]:
        pathP = '{}/parsed/{}'.format(SEMA3D_PATH, n)
        pathD = '{}/features/{}'.format(SEMA3D_PATH, n)
        pathC = '{}/superpoint_graphs/{}'.format(SEMA3D_PATH, n)
        if not os.path.exists(pathP):
            os.makedirs(pathP)
        random.seed(0)

        for file in os.listdir(pathC):
            print(file)
            if file.endswith(".h5"):
                f = h5py.File(pathD + file, 'r')
                xyz = f['xyz'][:]
                #rgb = f['rgb'][:].astype(np.float)
                rgb = np.empty_like(xyz)
                elpsv = np.stack([ f['xyz'][:,2][:], f['geof'][:,0][:], f['geof'][:,1][:], f['geof'][:,2][:], f['geof'][:,3][:] ], axis=1)

                # rescale to [-0.5,0.5]; keep xyz
                #warning - to use the trained model, make sure the elevation is comparable
                #to the set they were trained on
                #i.e. ~0 for roads and ~0.2-0.3 for builings for sema3d
                # and -0.5 for floor and 0.5 for ceiling for s3dis
                elpsv[:,0] /= 100 # (rough guess) #adapt 
                elpsv[:,1:] -= 0.5
                #rgb = rgb/255.0 - 0.5

                P = np.concatenate([xyz, rgb, elpsv], axis=1)
                f = h5py.File(pathC + file, 'r')
                numc = len(f['components'].keys())

                with h5py.File(pathP + file, 'w') as hf:
                    for c in range(numc):
                        idx = f['components/{:d}'.format(c)][:].flatten()
                        if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--PARISLILLE3D_PATH', default='datasets/ParisLille-3D')
    args = parser.parse_args()
    preprocess_pointclouds(args.PARISLILLE3D_PATH)


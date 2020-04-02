# Code to run SPG on ParisLille3D

You can find information relative to the dataset and the associated benchmark [here](https://npm3d.fr/)
We expect the following folder tree:

|superpoint_graph
    |learning
    |partition
    |datasets
        |ParisLille-3D
            |data
                |pointclouds
                    |ajaccio_2.ply
                        .
                        .
                        .
                    |Paris.ply

* Preprocessing as point clouds are too big

python utils/crop.py --n_train 5 --n_test 5 --ROOT_PATH datasets/ParisLille-3D

* Make partition

python partition/partition.py --dataset ParisLille3D --ROOT_PATH datasets/ParisLille-3D --reg_strength 0.8 --ver_batch 5000000 --voxel_width 0.03

* Prepare dataset

python learning/ParisLille3D_dataset.py --PARISLILLE3D_PATH datasets/ParisLille-3D

* Train from scratch (we assume a GPU is available)

CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset ParisLille3D --epochs 150 --lr_steps '[35, 70, 105]' --test_nth_epoch 300 --model_config 'gru_10,f_10' --pc_attribs xyzelpsv --odir "results/ParisLille-3D/trainval_best" --lr 1e-2 --val_split 0.4

* Inference on test set (we assume a GPU is available)

CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset ParisLille3D --model_config 'gru_10,f_10' --pc_attribs xyzelpsv --odir "results/ParisLille-3D/trainval_best" --lr 1e-2 --infer 1

* Oversample to get labels for the point cloud before undersampling

python partition/write_ParisLille3D.py --odir "results/ParisLille-3D/trainval_best" 

python utils/crop.py --n_train 5 --n_test 5 --ROOT_PATH datasets/ParisLille-3D

python partition/partition.py --dataset ParisLille3D --ROOT_PATH datasets/ParisLille-3D --reg_strength 0.8 --ver_batch 5000000 --voxel_width 0.01

python learning/sema3d_dataset.py --PARISLILLE3D_PATH datasets/ParisLille-3D

CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset ParisLille3D --epochs 150 --lr_steps '[35, 70, 105]' --test_nth_epoch 300 --model_config 'gru_10,f_10' --pc_attribs xyzelpsv --odir "results/ParisLille-3D/trainval_best" --lr 1e-2 --val_split 0.4



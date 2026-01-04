#!/bin/bash
mkdir -p checkpoints

# stage 1
python -u main.py --data_dir /KITTI/data/data_scene_flow \
    --stage kitti_s1 --restore_flow_ckpt model/raft.pth --restore_disp_ckpt model/aanet.pth \
    --gpus 0 1 2 --num_steps 100000 --batch_size 16 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision

# stage 2 
python -u main.py --data_dir /KITTI/data/data_scene_flow \
    --stage kitti_s2 --restore_flow_ckpt model/raft.pth --restore_disp_ckpt model/aanet.pth \
    --restore_pose_encoder_ckpt model/pose/pose_encoder.pth --restore_pose_decoder_ckpt model/pose/pose.pth \
    --gpus 0 1 --num_steps 100000 --batch_size 12 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision


# stage 3 
python -u main.py --data_dir /KITTI/data/data_scene_flow \
    --stage kitti_s3 --restore_flow_ckpt model/raft.pth --restore_disp_ckpt model/aanet.pth \
    --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0001 --image_size 375 1242 --wdecay 0.00001 --gamma=0.85 --mixed_precision


# stage 4 
python -u main.py --data_dir /Your/Dataset \
    --stage kitti_foggy --restore_flow_ckpt model/raft.pth --restore_flow_synimg_ckpt model/raft_synfog.pth \
    --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.00002 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision


# stage 5
python -u main.py --data_dir /Your/Dataset \
    --stage kitti_rain --restore_flow_ckpt model/raft.pth --restore_flow_synimg_rain_ckpt model/raft_synrain.pth \
    --gpus 0 1 --num_steps 100000 --batch_size 8 --lr 0.00001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --use_contra_loss --mixed_precision

# stage 6
python -u main.py --data_dir /Your/Dataset \
    --stage real_foggy --restore_flow_synimg_ckpt model/raft_synfog.pth --restore_flow_realimg_ckpt model/ch2da.pth \
    --gpus 0 --num_steps 100000 --batch_size 16 --lr 0.00001 --image_size 288 640 --wdecay 0.00001 --gamma=0.85 --mixed_precision

# stage 7
python -u main.py --data_dir /Your/Dataset \
    --stage real_rain --restore_flow_synimg_ckpt model/raft_synrain.pth --restore_flow_realimg_ckpt model/ch2da.pth \
    --gpus 0 --num_steps 100000 --batch_size 16 --lr 0.00001 --image_size 288 640 --wdecay 0.00001 --gamma=0.85 --mixed_precision

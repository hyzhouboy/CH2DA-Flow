from __future__ import print_function, division
import sys
sys.path.append('core')
import torch
import torch.nn.functional as F

import skimage.io
import argparse
import numpy as np
import os
import math

from aanet import AANet
from utils import transforms
from glob import glob

from numpy import savez_compressed
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/SceneFlow', type=str, help='Training dataset')

parser.add_argument('--stage', help="determines which dataset to use for training") 

parser.add_argument('--restore_flow_ckpt', help="restore checkpoint")
parser.add_argument('--restore_disp_ckpt', help="restore checkpoint")
parser.add_argument('--restore_pose_encoder_ckpt', help="restore checkpoint")
parser.add_argument('--restore_pose_decoder_ckpt', help="restore checkpoint")

# syn foggy
parser.add_argument('--restore_flow_synimg_ckpt', help="restore checkpoint")

parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--validation', type=str, nargs='+')

parser.add_argument('--lr', type=float, default=0.00002)
parser.add_argument('--num_steps', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
parser.add_argument('--val_batch_size', default=64, type=int, help='Batch size for validation')

# camera size
parser.add_argument('--camera_size', type=int, nargs='+', default=[375, 1242])

parser.add_argument('--gpus', type=int, nargs='+', default=[0])
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

# RAFT
parser.add_argument('--iters', type=int, default=12)
parser.add_argument('--wdecay', type=float, default=.00005)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')

# AANet
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')
parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')

parser.add_argument('--feature_similarity', default='correlation', type=str,
                    help='Similarity measure for matching cost')
parser.add_argument('--num_downsample', default=2, type=int, help='Number of downsample layer for feature extraction')
parser.add_argument('--num_scales', default=3, type=int, help='Number of stages when using parallel aggregation')

parser.add_argument('--load_pseudo_gt', action='store_true', help='Load pseudo gt for supervision')

# PoseNet
parser.add_argument('--num_layers', default=18, type=int, choices=[18, 34, 50, 101, 152], help='number of resnet layers')
parser.add_argument('--pose_model_type', default='seperate_pose', type=str,
                    help='Similarity measure for matching cost')

# disp-to-depth
parser.add_argument('--min_depth', default=0.1, type=float, help='minimum depth')
parser.add_argument('--max_depth', default=100.0, type=float, help='maximum depth')
args = parser.parse_args()
args.output_dir = 'pred_disp/'


def main():
    # For reproducibility
    if not os.path.isdir('generate_images'):
            os.mkdir('generate_images')
        
    if not os.path.isdir('pred_depth'):
        os.mkdir('pred_depth')
    
    if not os.path.isdir('pred_disp'):
        os.mkdir('pred_disp')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test loader
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    aanet = AANet(args)
    # model_disp = nn.DataParallel(AANet(self.args), device_ids=args.gpus)
    if os.path.exists(args.restore_disp_ckpt):
        print('=> Loading pretrained AANet:', args.restore_disp_ckpt)
        aanet.load_state_dict(torch.load(args.restore_disp_ckpt), strict=False)
    else:
        print('=> Using random initialization')

    if torch.cuda.device_count() >= 1:
        print('=> Use %d GPUs' % torch.cuda.device_count())
        aanet = torch.nn.DataParallel(aanet)

    # Inference
    aanet.cuda()
    aanet.eval()

    if args.data_dir.endswith('/'):
        args.data_dir = args.data_dir[:-1]

    # all_samples = sorted(glob(args.data_dir + '/*left.png'))
    all_samples = sorted(glob(args.data_dir + '/training/image_2/*.png'))

    num_samples = len(all_samples)
    print('=> %d samples found in the data dir' % num_samples)

    for i, sample_name in enumerate(all_samples):
        if i % 100 == 0:
            print('=> Inferencing %d/%d' % (i, num_samples))

        left_name = sample_name

        right_name = left_name.replace('image_2', 'image_3')

        left = np.array(Image.open(left_name).convert('RGB')).astype(np.float32)
        right = np.array(Image.open(right_name).convert('RGB')).astype(np.float32)
        sample = {'left': left,
                  'right': right}
        sample = test_transform(sample)  # to tensor and normalize

        left = sample['left'].cuda()  # [3, H, W]
        left = left.unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].cuda()
        right = right.unsqueeze(0)

        # Pad
        ori_height, ori_width = left.size()[2:]

        # Automatic
        factor = 48 
        args.img_height = math.ceil(ori_height / factor) * factor
        args.img_width = math.ceil(ori_width / factor) * factor

        if ori_height < args.img_height or ori_width < args.img_width:
            top_pad = args.img_height - ori_height
            right_pad = args.img_width - ori_width

            # Pad size: (left_pad, right_pad, top_pad, bottom_pad)
            left = F.pad(left, (0, right_pad, top_pad, 0))
            right = F.pad(right, (0, right_pad, top_pad, 0))

        with torch.no_grad():
            pred_disp = aanet(left, right)[-1]  # [B, H, W]

        if pred_disp.size(-1) < left.size(-1):
            pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
            pred_disp = F.interpolate(pred_disp, (left.size(-2), left.size(-1)),
                                      mode='bilinear') * (left.size(-1) / pred_disp.size(-1))
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]

        # Crop
        if ori_height < args.img_height or ori_width < args.img_width:
            if right_pad != 0:
                pred_disp = pred_disp[:, top_pad:, :-right_pad]
            else:
                pred_disp = pred_disp[:, top_pad:]

        disp = pred_disp[0].detach().cpu().numpy()  # [H, W]

        save_name = os.path.basename(left_name)
        save_name = os.path.join(args.output_dir, save_name)

        # if args.save_type == 'pfm':
        #     if args.visualize:
        skimage.io.imsave(save_name, (disp * 256.).astype(np.uint16))


if __name__ == '__main__':
    main()

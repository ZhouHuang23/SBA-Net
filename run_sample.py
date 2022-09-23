from core.libs import pyutils
import argparse
import torch
import numpy as np
import random
import os


def fix_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.set_deterministic(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=5, type=int)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    # parser.add_argument("--cam_crop_size", default=512, type=int)
    # parser.add_argument("--cam_batch_size", default=16, type=int)
    # parser.add_argument("--cam_crop_size", default=352, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=2, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.11, type=float)
    parser.add_argument("--cam_scales", default=(1.0,),
                        help="Multi-scale inferences")

    parser.add_argument("--conf_fg_thres", default=0.20, type=float)  # default=0.30
    parser.add_argument("--conf_bg_thres", default=0.07, type=float)

    # parameters for 'make_boundary_label'
    parser.add_argument("--window_size", default=13, type=int)
    parser.add_argument("--theta_scale", default=0.30, type=float)
    parser.add_argument("--theta_diff", default=0.10, type=float)
    parser.add_argument("--boundary_labels",
                        default={'BG': 0, 'FG': 50, 'BOUNDARY_FG_FG': 100, 'BOUNDARY_FG_BG': 150, 'IGNORE': 200},
                        type=dict)

    # Output Path
    parser.add_argument("--log_name", default="record", type=str)
    parser.add_argument("--cam_weights_name", default="outputs/weights/sess/res50_cam_attention.pth", type=str)
    parser.add_argument("--bes_weights_name", default="outputs/weights/sess/res50_bes.pth", type=str)
    parser.add_argument("--cam_out_dir", default="test_results/cam", type=str)
    parser.add_argument("--cam_vis_dir", default="test_results/cam_vis", type=str)
    parser.add_argument("--boundary_label_dir", default="test_results/boundary_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="test_results/sem_seg", type=str)

    # Step
    parser.add_argument("--train_cam_pass", default=False)
    parser.add_argument("--make_cam_pass", default=False)
    parser.add_argument("--make_boundary_label_pass", default=True)

    args = parser.parse_args()

    fix_seed(args.seed)
    os.makedirs("outputs/weight/sess", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.cam_vis_dir, exist_ok=True)
    os.makedirs(args.boundary_label_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)

    print(vars(args))

    if args.train_cam_pass is True:
        from core.train import train_cam

        timer = pyutils.Timer('step.train_cam:')
        train_cam.run(args)

    if args.make_cam_pass is True:
        from core.train import make_cam

        timer = pyutils.Timer('step.make_cam:')
        make_cam.run(args)

    if args.make_boundary_label_pass is True:
        from core.train import make_boundary_label

        timer = pyutils.Timer('step.cam_to_boundary_label:')
        make_boundary_label.run(args)

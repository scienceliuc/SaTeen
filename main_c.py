from logging import debug
import os

import time
import argparse
import json
import random
import numpy as np
from pycm import *

import pickle
from collections import defaultdict

import math
from typing import ValuesView

from dataset.selectedRotateImageFolder import prepare_test_data
from utils.utils import get_logger
from utils.cli_utils import *

import torch
import torch.nn.functional as F

import tent, eata, sar, deyo, recap_plpd, sateen
from sam import SAM

import models.Res as Resnet
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import _load_weights


def validate(val_loader, model, criterion, args, mode='eval'):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            # compute output
            output = model(images)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            if i > 10 and args.debug:
                break
    return top1.avg, top5.avg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description='ReCAP exps')

    # path
    parser.add_argument('--data', default='./data/imagenet', help='path to dataset')
    parser.add_argument('--data_corruption', default='./data/imagenet-c', help='path to corruption dataset')
    parser.add_argument('--output', default='./exps', help='the output directory of this experiment')

    parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training.')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')

    # dataloader
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--test_batch_size', default=64, type=int,
                        help='mini-batch size for testing, before default value is 4')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')

    # corruption settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')

    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int,
                        help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000.,
                        help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000) * 0.40,
                        help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05,
                        help='\epsilon in Eqn. (5) for filtering redundant samples')

    # Exp Settings
    parser.add_argument('--method', default='recap_plpd', type=str, help='no_adapt, tent, eata, sar, deyo, recap_plpd')
    parser.add_argument('--model', default='resnet50_gn_timm', type=str,
                        help='resnet50_gn_timm or resnet50_bn_torch or vitbase_timm')
    parser.add_argument('--exp_type', default='label_shifts', type=str, help='normal, mix_shifts, bs1, label_shifts')

    # SAR parameters
    parser.add_argument('--sar_margin_e0', default=0.4, type=float,
                        help='the threshold for reliable minimization in SAR, Eqn. (2)')
    parser.add_argument('--imbalance_ratio', default=500000, type=float,
                        help='imbalance ratio for label shift exps, selected from [1, 1000, 2000, 3000, 4000, 5000, 500000], 1  denotes totally uniform and 500000 denotes (almost the same to Pure Class Order).')

    # DeYO parameters
    parser.add_argument('--aug_type', default='patch', type=str, help='patch, pixel, occ')
    parser.add_argument('--occlusion_size', default=112, type=int)
    parser.add_argument('--row_start', default=56, type=int)
    parser.add_argument('--column_start', default=56, type=int)
    parser.add_argument('--deyo_margin', default=0.5, type=float,
                        help='Entropy threshold for sample selection $\tau_\mathrm{Ent}$ in Eqn. (8)')
    parser.add_argument('--deyo_margin_e0', default=0.2, type=float,
                        help='Entropy margin for sample weighting $\mathrm{Ent}_0$ in Eqn. (10)')
    parser.add_argument('--plpd_threshold', default=0.2, type=float,
                        help='PLPD threshold for sample selection $\tau_\mathrm{PLPD}$ in Eqn. (8)')
    parser.add_argument('--patch_len', default=4, type=int, help='The number of patches per row/column')
    parser.add_argument('--fishers', default=0, type=int)
    parser.add_argument('--filter_ent', default=1, type=int)
    parser.add_argument('--filter_plpd', default=1, type=int)
    parser.add_argument('--reweight_ent', default=1, type=int)
    parser.add_argument('--reweight_plpd', default=1, type=int)
    parser.add_argument('--topk', default=1000, type=int)

    # ReCAP parameters
    parser.add_argument('--weight_lr', default=1.0, type=float)
    parser.add_argument('--recap_margin', default=0.8, type=float,
                        help='Regional-Entropy threshold \tau_RE for sample selection in Eqn. (9); only samples with L_RE(x) < \tau_RE contribute to adaptation.')
    parser.add_argument('--recap_margin_L0', default=0.7, type=float,
                        help='Reference entropy L_0 in Eqn. (9) that converts Regional-Entropy to the weighting coefficient \alpha(x) = 1 / exp(L_RE(x) - L_0).')
    parser.add_argument('--weight_tau', default=1.2, type=float,
                        help='Region scale \tau in Eqn. (4); enlarges or shrinks the Gaussian neighborhood ')
    parser.add_argument('--weight_reg', default=0.5, type=float,
                        help='Trade-off \lambda in Eqn. (9) between Regional Entropy L_RE and Regional Instability ')
    parser.add_argument('--reweight_threshold', default=2.0, type=float,
                        help='Upper bound for the re-weighting coefficient \alpha(x); clips extreme values to avoid gradient explosion during adaptation.')

    # SaTeen parameters
    parser.add_argument('--lambda_1', default=0.4, type=float)
    parser.add_argument('--lambda_2', default=0.05, type=float)
    parser.add_argument('--sateen_margin', default=0.2, type=float)
    parser.add_argument('--k', default=64, type=int)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    if torch.cuda.is_available() and args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    args.num_class = 1000

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # set random seeds
    if args.seed is not None:
        set_seed(args.seed)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    args.logger_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + "-{}-{}-level{}-seed{}.txt".format(
        args.method, args.model, args.level, args.seed)

    logger = get_logger(name="project", output_directory=args.output, log_name=args.logger_name, debug=False)

    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
                          'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
                          'pixelate', 'jpeg_compression']

    if args.exp_type == 'mix_shifts':
        datasets = []
        for cpt in common_corruptions:
            args.corruption = cpt
            logger.info(args.corruption)

            val_dataset, _ = prepare_test_data(args)
            if args.method in ['tent', 'no_adapt', 'eata', 'sar', 'deyo', 'recap_plpd', 'sateen']:
                val_dataset.switch_mode(True, False)
            else:
                assert False, NotImplementedError
            datasets.append(val_dataset)

        from torch.utils.data import ConcatDataset

        mixed_dataset = ConcatDataset(datasets)
        logger.info(f"length of mixed dataset us {len(mixed_dataset)}")

        val_loader = torch.utils.data.DataLoader(
            mixed_dataset,
            batch_size=args.test_batch_size,
            shuffle=args.if_shuffle,
            num_workers=args.workers,
            pin_memory=True)

        common_corruptions = ['mix_shifts']

    args.e_margin *= math.log(args.num_class)
    args.sar_margin_e0 *= math.log(args.num_class)
    args.deyo_margin *= math.log(args.num_class)
    args.deyo_margin_e0 *= math.log(args.num_class)
    args.sateen_margin *= math.log(args.num_class)

    if args.model == 'resnet50_gn_timm':
        args.lambda_1 = 0.4
        args.lambda_2 = 0.05
        args.sateen_threshold = 0.2 * math.log(args.num_class)
    elif args.model == 'vitbase_timm':
        args.lambda_1 = 0.2
        args.lambda_2 = 0.05
        args.sateen_threshold = 0.35 * math.log(args.num_class)
    else:
        assert False, NotImplementedError

    args.recap_margin *= math.log(args.num_class)
    args.recap_margin_L0 *= math.log(args.num_class)
    args.sigmas = torch.from_numpy(np.load(f'utils/cov_{args.model}.npy'))

    if args.exp_type == 'bs1':
        args.reweight_threshold = 5.0
        args.test_batch_size = 1
        logger.info("modify batch size to 1, for exp of single sample adaptation")

    if args.exp_type == 'label_shifts':
        args.if_shuffle = False
        logger.info("this exp is for label shifts, no need to shuffle the dataloader, use our pre-defined sample order")

    acc1s, acc5s = [], []
    ir = args.imbalance_ratio

    bs = args.test_batch_size
    args.print_freq = 50000 // 20 // bs

    # build model for adaptation
    if args.method in ['tent', 'eata', 'sar', 'no_adapt', 'deyo', 'recap_plpd', 'sateen']:
        if args.model == "resnet50_gn_timm":
            checkpoint = torch.load('./models/resnet50_gn_a1h2-8fe6c4d0.pth')
            net = timm.create_model('resnet50_gn', pretrained=False)
            net.load_state_dict(checkpoint)
            # net = timm.create_model('resnet50_gn', pretrained=True)
            print("Model created successfully!")
            args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.0001

        elif args.model == "vitbase_timm":
            npz_path = './models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'
            net = timm.create_model('vit_base_patch16_224', pretrained=False)
            _load_weights(net, npz_path)
            # net = timm.create_model('vit_base_patch16_224', pretrained=True)
            print("Model created successfully!")
            args.lr = (0.0005 / 64) * bs
        elif args.model == "resnet50_bn_torch":
            net = Resnet.__dict__['resnet50'](pretrained=False)
            init = torch.load("./models/resnet50-19c8e357.pth")
            net.load_state_dict(init)
            # net = Resnet.__dict__['resnet50'](pretrained=True)
            print("Model created successfully!")
            args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00003
        else:
            assert False, NotImplementedError
        net = net.cuda()
        args.lr = args.lr * args.weight_lr
    else:
        assert False, NotImplementedError



    if args.exp_type == 'bs1' and args.method == 'sar':
        args.lr = 2 * args.lr
        logger.info("double lr for sar under bs=1")

    if args.exp_type == 'bs1' and args.method == 'deyo':
        args.lr = 2 * args.lr
        logger.info("double lr for deyo under bs=1")

    if args.exp_type == 'bs1' and ('recap' in args.method):
        args.lr = 2 * args.lr
        logger.info("double lr for recap under bs=1")

    for corrupt in common_corruptions:
        args.corruption = corrupt



        if args.method in ['tent', 'eata', 'sar', 'no_adapt', 'deyo', 'recap_plpd', 'sateen']:
            if args.corruption != 'mix_shifts':
                val_dataset, val_loader = prepare_test_data(args)
                val_dataset.switch_mode(True, False)
        else:
            assert False, NotImplementedError

        # construt new dataset with online imbalanced label distribution shifts
        # note that this operation does not support mix-domain-shifts exps
        if args.exp_type == 'label_shifts':
            logger.info(f"imbalance ratio is {ir}")
            if args.seed == 2024:
                indices_path = './dataset/total_{}_ir_{}_class_order_shuffle_yes.npy'.format(100000, ir)
            else:
                indices_path = './dataset/seed{}_total_{}_ir_{}_class_order_shuffle_yes.npy'.format(args.seed, 100000,
                                                                                                    ir)
            logger.info(f"label_shifts_indices_path is {indices_path}")
            indices = np.load(indices_path)
            val_dataset.set_specific_subset(indices.astype(int).tolist())



        set_seed(args.seed)

        if args.method == "tent":
            net = tent.configure_model(net)
            params, param_names = tent.collect_params(net)
            # logger.info(param_names)
            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            tented_model = tent.Tent(net, optimizer)

            top1, top5 = validate(val_loader, tented_model, None, args, mode='eval')
            logger.info(
                f"Result under {args.corruption}. The adapttion accuracy of Tent is top1 {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method == "no_adapt":
            tented_model = net
            top1, top5 = validate(val_loader, tented_model, None, args, mode='eval')
            logger.info(
                f"Result under {args.corruption}. Original Accuracy (no adapt) is top1: {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method == "eata":
            # compute fisher informatrix
            args.corruption = 'original'
            fisher_dataset, fisher_loader = prepare_test_data(args)
            fisher_dataset.set_dataset_size(args.fisher_size)
            fisher_dataset.switch_mode(True, False)

            net = eata.configure_model(net)
            params, param_names = eata.collect_params(net)
            # fishers = None
            ewc_optimizer = torch.optim.SGD(params, 0.001)
            fishers = {}
            train_loss_fn = nn.CrossEntropyLoss().cuda()
            for iter_, (images, targets) in enumerate(fisher_loader, start=1):
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    targets = targets.cuda(args.gpu, non_blocking=True)
                outputs = net(images)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in net.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == len(fisher_loader):
                            fisher = fisher / iter_
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()
            logger.info("compute fisher matrices finished")
            del ewc_optimizer

            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            adapt_model = eata.EATA(net, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin,
                                    d_margin=args.d_margin)

            top1, top5 = validate(val_loader, adapt_model, None, args, mode='eval')
            logger.info(
                f"Result under {args.corruption}. After EATA Adapt: Accuracy: top1: {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method in ['sar']:
            net = sar.configure_model(net)
            params, param_names = sar.collect_params(net)
            logger.info(param_names)

            base_optimizer = torch.optim.SGD
            optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
            adapt_model = sar.SAR(net, optimizer, margin_e0=args.sar_margin_e0)

            batch_time = AverageMeter('Time', ':6.3f')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            progress = ProgressMeter(
                len(val_loader),
                [batch_time, top1, top5],
                prefix='Test: ')
            end = time.time()
            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                if args.gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    target = target.cuda()
                output = adapt_model(images)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

            acc1 = top1.avg
            acc5 = top5.avg

            logger.info(
                f"Result under {args.corruption}. The adaptation accuracy of SAR is top1: {acc1:.5f} and top5: {acc5:.5f}")

            acc1s.append(top1.avg.item())
            acc5s.append(top5.avg.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method in ['deyo']:

            biased = (args.exp_type == 'spurious')

            net = deyo.configure_model(net)
            params, param_names = deyo.collect_params(net)
            logger.info(param_names)

            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            adapt_model = deyo.DeYO(net, args, optimizer, deyo_margin=args.deyo_margin, margin_e0=args.deyo_margin_e0)

            batch_time = AverageMeter('Time', ':6.3f')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            if biased:
                LL_AM = AverageMeter('LL Acc', ':6.2f')
                LS_AM = AverageMeter('LS Acc', ':6.2f')
                SL_AM = AverageMeter('SL Acc', ':6.2f')
                SS_AM = AverageMeter('SS Acc', ':6.2f')
                progress = ProgressMeter(
                    len(val_loader),
                    [batch_time, top1, top5, LL_AM, LS_AM, SL_AM, SS_AM],
                    prefix='Test: ')
            else:
                progress = ProgressMeter(
                    len(val_loader),
                    [batch_time, top1, top5],
                    prefix='Test: ')

            end = time.time()
            count_backward = 1e-6
            final_count_backward = 1e-6
            count_corr_pl_1 = 0
            count_corr_pl_2 = 0
            total_count_backward = 1e-6
            total_final_count_backward = 1e-6
            total_count_corr_pl_1 = 0
            total_count_corr_pl_2 = 0
            correct_count = [0, 0, 0, 0]
            total_count = [1e-6, 1e-6, 1e-6, 1e-6]

            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                if args.gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    target = target.cuda()
                if biased:
                    place = dl[2].cuda()
                    group = 2 * target + place
                else:
                    group = None

                output, backward, final_backward, corr_pl_1, corr_pl_2 = adapt_model(images, i, target, group=group)
                if biased:
                    TFtensor = (output.argmax(dim=1) == target)

                    for group_idx in range(4):
                        correct_count[group_idx] += TFtensor[group == group_idx].sum().item()
                        total_count[group_idx] += len(TFtensor[group == group_idx])
                    acc1, acc5 = accuracy(output, target, topk=(1, 1))
                else:
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))

                count_backward += backward
                final_count_backward += final_backward
                total_count_backward += backward
                total_final_count_backward += final_backward

                count_corr_pl_1 += corr_pl_1
                count_corr_pl_2 += corr_pl_2
                total_count_corr_pl_1 += corr_pl_1
                total_count_corr_pl_2 += corr_pl_2

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                if i % args.print_freq == 0:
                    if biased:
                        LL = correct_count[0] / total_count[0] * 100
                        LS = correct_count[1] / total_count[1] * 100
                        SL = correct_count[2] / total_count[2] * 100
                        SS = correct_count[3] / total_count[3] * 100
                        LL_AM.update(LL, images.size(0))
                        LS_AM.update(LS, images.size(0))
                        SL_AM.update(SL, images.size(0))
                        SS_AM.update(SS, images.size(0))

                    count_backward = 1e-6
                    final_count_backward = 1e-6
                    count_corr_pl_1 = 0
                    count_corr_pl_2 = 0

                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

            acc1 = top1.avg
            acc5 = top5.avg

            logger.info(
                f"Result under {args.corruption}. The adaptation accuracy of DeYO is top1: {acc1:.5f} and top5: {acc5:.5f}")

            acc1s.append(top1.avg.item())
            acc5s.append(top5.avg.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method in ['recap_plpd']:

            net = recap_plpd.configure_model(net)
            params, param_names = recap_plpd.collect_params(net)
            logger.info(param_names)

            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            adapt_model = recap_plpd.ReCAP(net, optimizer, \
                                           margin=args.recap_margin, \
                                           margin_L0=args.recap_margin_L0, \
                                           weight_reg=args.weight_reg, reweight_threshold=args.reweight_threshold, \
                                           sigmas=args.sigmas, batch_size=bs, weight_tau=args.weight_tau)

            batch_time = AverageMeter('Time', ':6.3f')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            progress = ProgressMeter(
                len(val_loader),
                [batch_time, top1, top5],
                prefix='Test: ')

            end = time.time()
            start_time = time.time()

            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                if args.gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    target = target.cuda()

                output = adapt_model(images)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # measure elapsed time

                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

            acc1 = top1.avg
            acc5 = top5.avg

            logger.info(
                f"Result under {args.corruption}. The adaptation accuracy of ReCAP+plpd is top1: {acc1:.5f} and top5: {acc5:.5f}")

            acc1s.append(top1.avg.item())
            acc5s.append(top5.avg.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")
        elif args.method in ['sateen']:

            biased = (args.exp_type == 'spurious')

            net = deyo.configure_model(net)
            params, param_names = deyo.collect_params(net)
            logger.info(param_names)

            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            adapt_model = sateen.SaTeen(net, args, optimizer, sateen_margin=args.sateen_margin)

            batch_time = AverageMeter('Time', ':6.3f')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            if biased:
                LL_AM = AverageMeter('LL Acc', ':6.2f')
                LS_AM = AverageMeter('LS Acc', ':6.2f')
                SL_AM = AverageMeter('SL Acc', ':6.2f')
                SS_AM = AverageMeter('SS Acc', ':6.2f')
                progress = ProgressMeter(
                    len(val_loader),
                    [batch_time, top1, top5, LL_AM, LS_AM, SL_AM, SS_AM],
                    prefix='Test: ')
            else:
                progress = ProgressMeter(
                    len(val_loader),
                    [batch_time, top1, top5],
                    prefix='Test: ')

            end = time.time()
            count_backward = 1e-6
            final_count_backward = 1e-6
            count_corr_pl_1 = 0
            count_corr_pl_2 = 0
            total_count_backward = 1e-6
            total_final_count_backward = 1e-6
            total_count_corr_pl_1 = 0
            total_count_corr_pl_2 = 0
            correct_count = [0, 0, 0, 0]
            total_count = [1e-6, 1e-6, 1e-6, 1e-6]

            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                if args.gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    target = target.cuda()
                if biased:
                    place = dl[2].cuda()
                    group = 2 * target + place
                else:
                    group = None

                output, backward, final_backward, corr_pl_1, corr_pl_2 = adapt_model(images, i, target, group=group)
                if biased:
                    TFtensor = (output.argmax(dim=1) == target)

                    for group_idx in range(4):
                        correct_count[group_idx] += TFtensor[group == group_idx].sum().item()
                        total_count[group_idx] += len(TFtensor[group == group_idx])
                    acc1, acc5 = accuracy(output, target, topk=(1, 1))
                else:
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))

                count_backward += backward
                final_count_backward += final_backward
                total_count_backward += backward
                total_final_count_backward += final_backward

                count_corr_pl_1 += corr_pl_1
                count_corr_pl_2 += corr_pl_2
                total_count_corr_pl_1 += corr_pl_1
                total_count_corr_pl_2 += corr_pl_2

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                if i % args.print_freq == 0:
                    if biased:
                        LL = correct_count[0] / total_count[0] * 100
                        LS = correct_count[1] / total_count[1] * 100
                        SL = correct_count[2] / total_count[2] * 100
                        SS = correct_count[3] / total_count[3] * 100
                        LL_AM.update(LL, images.size(0))
                        LS_AM.update(LS, images.size(0))
                        SL_AM.update(SL, images.size(0))
                        SS_AM.update(SS, images.size(0))

                    count_backward = 1e-6
                    final_count_backward = 1e-6
                    count_corr_pl_1 = 0
                    count_corr_pl_2 = 0

                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

            acc1 = top1.avg
            acc5 = top5.avg

            logger.info(
                f"Result under {args.corruption}. The adaptation accuracy of DeYO is top1: {acc1:.5f} and top5: {acc5:.5f}")

            acc1s.append(top1.avg.item())
            acc5s.append(top5.avg.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        else:
            assert False, NotImplementedError
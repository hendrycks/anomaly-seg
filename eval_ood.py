# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from config import cfg
from dataset import ValDataset
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm

import anom_utils

colors = loadmat('data/color150.mat')['colors']


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))

def eval_ood_measure(conf, seg_label, cfg, mask=None):
    out_labels = cfg.OOD.out_labels
    if mask is not None:
        seg_label = seg_label[mask]

    out_label = seg_label == out_labels[0]
    for label in out_labels:
        out_label = np.logical_or(out_label, seg_label == label)

    in_scores = - conf[np.logical_not(out_label)]
    out_scores  = - conf[out_label]

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = anom_utils.get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr
    else:
        print("This image does not contain any OOD pixels or is only OOD.")
        return None


def evaluate(segmentation_module, loader, cfg, gpu):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()

    aurocs, auprs, fprs = [], [], []

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            tmp_scores = scores
            if cfg.OOD.exclude_back:
                tmp_scores = tmp_scores[:,1:]


            mask = None
            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

            #for evaluating MSP
            if cfg.OOD.ood == "msp":
                conf, _  = torch.max(nn.functional.softmax(tmp_scores, dim=1),dim=1)
                conf = as_numpy(conf.squeeze(0).cpu())
            elif cfg.OOD.ood == "maxlogit":
                conf, _  = torch.max(tmp_scores,dim=1)
                conf = as_numpy(conf.squeeze(0).cpu())
            elif cfg.OOD.ood == "background":
                conf = tmp_scores[:, 0]
                conf = as_numpy(conf.squeeze(0).cpu())
            elif cfg.OOD.ood == "crf":
                import pydensecrf.densecrf as dcrf
                from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
                ch,h,w = scores.squeeze(0).size()
                d = dcrf.DenseCRF2D(h, w, ch)  # width, height, nlabels
                tmp_scores = as_numpy(nn.functional.softmax(tmp_scores, dim=1).squeeze(0))
                tmp_scores = as_numpy(tmp_scores)
                U = unary_from_softmax(tmp_scores)
                d.setUnaryEnergy(U)

                pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=13, img=tmp_scores, chdim=0)
                d.addPairwiseEnergy(pairwise_energy, compat=10)
                # Run inference for 100 iterations
                Q_unary = d.inference(100)
                # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
                map_soln_unary = np.argmax(Q_unary, axis=0)

                # Unfortunately, the DenseCRF flattens everything, so get it back into picture form.
                map_soln_unary = map_soln_unary.reshape((h,w))
                conf = np.max(Q_unary, axis=0).reshape((h,w))
            elif cfg.OOD.ood == "crf-gauss":
                import pydensecrf.densecrf as dcrf
                from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
                ch,h,w = scores.squeeze(0).size()
                d = dcrf.DenseCRF2D(h, w, ch)  # width, height, nlabels
                tmp_scores = as_numpy(nn.functional.softmax(tmp_scores, dim=1).squeeze(0))
                tmp_scores = as_numpy(tmp_scores)
                U = unary_from_softmax(tmp_scores)
                d.setUnaryEnergy(U)
                d.addPairwiseGaussian(sxy=3, compat=3)  # `compat` is the "strength" of this potential.

                # Run inference for 100 iterations
                Q_unary = d.inference(100)
                # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
                map_soln_unary = np.argmax(Q_unary, axis=0)

                # Unfortunately, the DenseCRF flattens everything, so get it back into picture form.
                map_soln_unary = map_soln_unary.reshape((h,w))
                conf = np.max(Q_unary, axis=0).reshape((h,w))
            res = eval_ood_measure(conf, seg_label, cfg, mask=mask)
            if res is not None:
                auroc, aupr, fpr = res
                aurocs.append(auroc); auprs.append(aupr), fprs.append(fpr)
            else:
                pass


        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result')
            )

        pbar.update(1)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))
    print("mean auroc = ", np.mean(aurocs), "mean aupr = ", np.mean(auprs), " mean fpr = ", np.mean(fprs))

def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        help="gpu to use"
    )
    parser.add_argument(
        "--ood",
        help="Choices are [msp, crf-gauss, crf, maxlogit, background]",
        default="msp",
    )
    parser.add_argument(
        "--exclude_back",
        help="Whether to exclude the background class.",
        action="store_true",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    ood = ["OOD.exclude_back", args.exclude_back, "OOD.ood", args.ood]

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(ood)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    main(cfg, args.gpu)

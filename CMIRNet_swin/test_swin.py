import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime
import os
import time
from skimage.transform import resize
import torch
import torch.utils.data
from torch import nn
# from bert.modeling_bert import BertModel
from transformers import BertModel
import torchvision

# from lib import segmentation

import transforms as T
from data.dataset_refer_bert import ReferDataset
from train_utils import distributed_utils as utils
import numpy as np
from imageio.v2 import imread, imsave
from PIL import Image
from pycocotools import mask
# from scipy.misc import imread
from PIL import Image
import swintransformer.segmentation as swin
from src.CMIRNet_swin import CMIRNet_swin


def get_dataset(image_set, transform, args):
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                    #   input_size=(256, 448),
                      eval_mode=True)

    return ds


def evaluate(args, model, data_loader, ref_ids, refer, bert_model, device, num_classes, display=False,
             baseline_model=None,
             objs_ids=None, num_objs_list=None):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    refs_ids_list = []
    outputs = []
    # dict to save results for DAVIS
    total_outputs = {}

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    header = 'Test:'
    with torch.no_grad():
        k = 0
        ll = 0
        out_txt = open('iou.txt','w+') 
        for image, target, sentences, attentions in metric_logger.log_every(data_loader, 100, header):

            image, target, sentences, attentions = image.to(device), target.to(device), sentences.to(
                device), attentions.to(device)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            target = target.cpu().data.numpy()

            for j in range(sentences.size(-1)):

                refs_ids_list.append(k)

                last_hidden_states = bert_model(sentences[:, :, j], attentions[:, :, j])[0]
                embedding = last_hidden_states[:, 0, :]
                l = last_hidden_states.permute(0,2,1)
                output,_,_ = model(image, l, attentions[:, :, j].unsqueeze(-1),embedding)

                output = output['out'].cpu()
                output_mask = output.argmax(1).data.numpy()
                if display:
                    outputs.append(output_mask)

                I, U = computeIoU(output_mask, target)

                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I * 1.0 / U

                mean_IoU.append(this_iou)
                print(this_iou,file=out_txt)

                cum_I += I
                cum_U += U

                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            del image, target, attentions

            if display:

                
                ref = refer.loadRefs(ref_ids[k])
                image_info = refer.Imgs[ref[0]['image_id']]
                results_folder = args.results_folder
                for p in range(len(ref[0]['sentences'])):
                    ll += 1

                    plt.figure()
                    plt.axis('off')
                    plt.xticks([])    # 去 x 轴刻度
                    plt.yticks([]) 

                    sentence = ref[0]['sentences'][p]['raw']
                    im_path = os.path.join(refer.IMAGE_DIR, image_info['file_name'])

                    im = imread(im_path)
                    
                    im = resize(im, (480, 480))
                    plt.imshow(im)

                    # plt.text(0, 0, sentence, fontsize=12)

                    ax = plt.gca()
                    ax.set_autoscale_on(False)

                    # mask definition
                    img = np.ones((im.shape[0], im.shape[1], 3))
                    color_mask = np.array([0, 255, 0]) / 255.0
                    for i in range(3):
                        img[:, :, i] = color_mask[i]

                    output_mask = outputs[-len(ref[0]['sentences']) + p].transpose(1, 2, 0)

                    ax.imshow(np.dstack((img, output_mask * 0.5)))

                    if not os.path.isdir(results_folder):
                        os.makedirs(results_folder)

                    this_iou = mean_IoU[-len(ref[0]['sentences']) + p]

                    

                    figname = os.path.join(results_folder, str(ll) + '.png')
                    f = plt.gcf()
                    f.savefig(figname, bbox_inches='tight', pad_inches=0.0,)
                    f.clear()

                    plt.close()

            k += 1
    out_txt.close()
    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)

    print(results_str)

    return refs_ids_list, outputs


def get_transform():
    transforms = []
    transforms.append(T.RandomResize(480))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


# compute IoU
def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def create_model(num_classes, args):
    Vis_backbone_TGMM = swin.swin_backbone_TGMM(pretrained=args.pretrained, args=args)
    base_model = CMIRNet_swin

    model = base_model(Vis_backbone_TGMM, num_classes, args)

    return model

def main(args):
    device = torch.device(args.device)
    num_classes = 2

    dataset_test = get_dataset(args.split, get_transform(), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=test_sampler,
                                                   num_workers=args.workers, collate_fn=utils.collate_fn_emb_berts)

    # model = segmentation.__dict__[args.model](
    #                                           args=args)
    model = create_model(num_classes=num_classes, args=args)

    model.to(device)
    model_class = BertModel

    bert_model = model_class.from_pretrained(args.ck_bert)
    bert_model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')

    bert_model.load_state_dict(checkpoint['bert_model'],  strict=False)
    model.load_state_dict(checkpoint['model'])

    if args.dataset == 'refcoco' or args.dataset == 'refcoco+' or args.dataset == 'refcocog':
        ref_ids = dataset_test.ref_ids
        refer = dataset_test.refer
        ids = ref_ids
        objs_ids = None
        num_objs_list = None

    baseline_model = None

    refs_ids_list, outputs = evaluate(args, model, data_loader_test, ids, refer, bert_model, device=device,
                                      num_classes=2, baseline_model=baseline_model, objs_ids=objs_ids,
                                      num_objs_list=num_objs_list, display=args.display)


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()
    main(args)

import torch
from torch import nn
import train_utils.distributed_utils_pretrain as utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

import numpy as np  

import torch  
  
def batch_bbox_iou_torch(x, y):  
    """  
    使用PyTorch Tensor来计算两个批次bbox的IoU。  
      
    参数:  
    x -- 第一个批次的bbox，形状为(B, 4)，其中B是批次大小，4是bbox的坐标(x1, y1, x2, y2)  
    y -- 第二个批次的bbox，形状与x相同  
      
    返回:  
    ious -- 一个形状为(B,)的Tensor，包含每个对应bbox对的IoU值  
    """  
    # 确保x和y是torch.Tensor  
    x = torch.tensor(x, dtype=torch.float)  
    y = torch.tensor(y, dtype=torch.float)  
      
    # 计算交集区域的坐标  
    xi1 = torch.max(x[:, 0], y[:, 0])  
    yi1 = torch.max(x[:, 1], y[:, 1])  
    xi2 = torch.min(x[:, 2], y[:, 2])  
    yi2 = torch.min(x[:, 3], y[:, 3])  
      
    # 计算交集区域的宽和高  
    inter_widths = torch.clamp(xi2 - xi1, min=0)  
    inter_heights = torch.clamp(yi2 - yi1, min=0)  
      
    # 计算交集区域的面积  
    inter_areas = inter_widths * inter_heights  
      
    # 计算两个bbox的面积  
    box1_areas = (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  
    box2_areas = (y[:, 2] - y[:, 0]) * (y[:, 3] - y[:, 1])  
      
    # 计算并集区域的面积  
    union_areas = box1_areas + box2_areas - inter_areas  
      
    # 计算IoU，注意避免除以零  
    eps = 1e-7  # 添加一个小的epsilon值以避免除以零  
    ious = inter_areas / (union_areas + eps)  
      
    return ious[0], inter_areas[0], union_areas[0]
    

def iou_loss(x, y):  
    """  
    尝试使用NumPy向量化操作来计算两个批次bbox的IoU，但注意这仍然需要内部遍历。  
      
    参数:  
    x -- 第一个批次的bbox，形状为(B, 4)，其中B是批次大小，4是bbox的坐标(x1, y1, x2, y2)  
    y -- 第二个批次的bbox，形状与x相同  
      
    返回:  
    ious -- 一个形状为(B,)的数组，包含每个对应bbox对的IoU值  
    """  
    # 计算交集区域的坐标  
    xi1 = torch.max(x[:, 0], y[:, 0])  
    yi1 = torch.max(x[:, 1], y[:, 1])  
    xi2 = torch.min(x[:, 2], y[:, 2])  
    yi2 = torch.min(x[:, 3], y[:, 3])  
      
    # 计算交集区域的宽和高  
    inter_widths = torch.clamp(xi2 - xi1, min=0)  
    inter_heights = torch.clamp(yi2 - yi1, min=0)  
      
    # 计算交集区域的面积  
    inter_areas = inter_widths * inter_heights  
      
    # 计算两个bbox的面积  
    box1_areas = (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  
    box2_areas = (y[:, 2] - y[:, 0]) * (y[:, 3] - y[:, 1])  
      
    # 计算并集区域的面积  
    union_areas = box1_areas + box2_areas - inter_areas  
      
    # 计算IoU，注意避免除以零  
    eps = 1e-7  # 添加一个小的epsilon值以避免除以零  
    ious = inter_areas / (union_areas + eps)  

    ious = 1 - ious
    iou_loss = torch.mean(ious)
      
    return iou_loss  
  

def criterion(inputs, target, device):
    losses = {}
    loss = 0.00
    total = 0
    weight = torch.FloatTensor([0.9, 1.1]).to(device)
    for name, x in inputs.items():

        losses[name] = nn.functional.cross_entropy(x, target, weight=weight, ignore_index=255)
        loss += losses[name]
        total += 1

    return loss / float(total)


# IoU calculation for proper validation
def IoU(pred, gt):
    pred = pred.argmax(1)
    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union


def find_bounding_boxes(masks):  
    """  
    找到每个mask的最小外接矩形。  
  
    参数:  
    - masks: 形状为[B, H, W]的NumPy数组，其中B是批次大小，H是高度，W是宽度。  
  
    返回:  
    - bounding_boxes: 形状为[B, 4]的NumPy数组，每个元素是一个包含[y_min, x_min, y_max, x_max]的矩形。  
    """  
    B, H, W = masks.shape  
    bounding_boxes = np.zeros((B, 4), dtype=np.int32)  
  
    for i in range(B):  
        mask = masks[i]  
        # 找到值为1的像素的坐标  
        y_coords, x_coords = np.where(mask == 1)  
          
        if len(y_coords) == 0:  # 如果没有值为1的像素，则边界设为全0  
            bounding_boxes[i] = [0, 0, 0, 0]  
        else:  
            # 计算最小外接矩形的边界  
            y_min, x_min = np.min(y_coords), np.min(x_coords)  
            y_max, x_max = np.max(y_coords), np.max(x_coords)  
            bounding_boxes[i] = [y_min, x_min, y_max, x_max]  
  
    return bounding_boxes  

def evaluate(model, data_loader, bert_model, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    with torch.no_grad():
        for image, target, sentences, attentions in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions = image.to(device), target.to(device), sentences.to(
                device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.squeeze(1).squeeze(1)

            total_its += 1

            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]

            embedding = last_hidden_states[:, 0, :]

            l_feat = last_hidden_states.permute(0, 2, 1)
            output, _, _ = model(image, l_feat, attentions.unsqueeze(-1), embedding)

            iou, I, U = batch_bbox_iou_torch(output["out"], target)
            acc_ious += iou
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)

            output = output['out']
            # confmat.update(target.flatten(), output.argmax(1).flatten())

        # confmat.reduce_from_all_processes()

        iou = acc_ious / total_its
        val_info = "mean IOU = %.2f\n" % (iou * 100.)
        for n_eval_iou in range(len(eval_seg_iou_list)):
            val_info += '    precision@%s = %.2f\n' % \
                           (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / total_its)
        val_info += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
        overallIOU = (cum_I * 100. / cum_U)
    return confmat, iou,overallIOU, val_info



def train_one_epoch(model, optimizer, data_loader, device, epoch, bert_model, lr_scheduler, val_loader, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    i = 0
    for image, target, sentences, attentions in metric_logger.log_every(data_loader, print_freq, header):
        image, target, sentences, attentions = image.to(device), target.to(device), sentences.to(device), attentions.to(device)
        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)
        target = target.squeeze(1).squeeze(1)
        last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
        embedding = last_hidden_states[:, 0, :]

        l_feat = last_hidden_states.permute(0, 2, 1)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output,_,_ = model(image, l_feat, attentions.unsqueeze(-1), embedding)

            loss = iou_loss(output["out"], target)
        
        # with open("./palette.json", "rb") as f:
        #     pallette_dict = json.load(f)
        #     pallette = []
        # for v in pallette_dict.values():
        #     pallette += v
        # grod = target[0,:,:]
        # grod = grod.to("cpu").numpy().astype(np.uint8)
        # grod = Image.fromarray(grod)
        # grod.putpalette(pallette)
    
        # prediction = output['out']

        # prediction = prediction.argmax(1)[0,:,:]
        
        # prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # mask = Image.fromarray(prediction)
        # mask.putpalette(pallette)
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(mask)
        # plt.subplot(1, 2, 2)
        # plt.imshow(grod)
        # plt.savefig(f'savefig_example_.png')
        # plt.show()
        # plt.close()
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)
        # i+=1
        # if i==2000:
        #     return metric_logger.meters["loss"].global_avg, lr

        # i+=1
        # if i==10:
        #     i=0
        #     confmat, iou,oIOU, val_info = evaluate(model, val_loader, bert_model, device=device, num_classes=2)
        #     print(val_info)
        #     print(oIOU)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            # alpha = float(x) / (warmup_epochs * num_step)
            # # warmup过程中lr倍率因子从warmup_factor -> 1
            # return warmup_factor * (1 - alpha) + alpha
            # alpha = float(x) / (warmup_epochs * num_step)
            # return (1 - alpha) + 0.2
            return -0.05 * (float(x) / num_step) + 1.0  #0轮对应0.00005,5轮对应0.00002，学习率线性下降
            # return -0.2 * (float(x) / num_step) + 1.4  #0轮对应0.00007,5轮对应0.00002，学习率线性下降
            # return -0.12 * (float(x) / num_step) + 1.2  # 0轮对应0.00006,5轮对应0.00003，学习率线性下降
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return 0.5 * (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9  # 5-40轮，从0.00002线性下降至0
            # return 0.6 * (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9  # 5-40轮，从0.00003线性下降至0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

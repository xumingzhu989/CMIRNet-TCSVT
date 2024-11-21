import operator
import os
import time
import datetime
from functools import reduce

from transformers import BertTokenizer, BertModel
import torch
from data.dataset_refer_bert import ReferDataset
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
import transforms as T
import train_utils.distributed_utils as utils
import resnet.resnet1 as res
from src.deeplabv3_emb import DeepLabV3Emb_512
from args import get_parser


def get_dataset(image_set, transform, args):
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      )

    return ds




def get_transform(train, base_size=520, crop_size=480, img_size=480):
    transforms = []
    if train:
        transforms.append(T.Resize(base_size))
        transforms.append(T.RandomCrop(crop_size))
    else:
        transforms.append(T.Resize(img_size))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)

def create_model(num_classes, args):
    # backbone = res.resnet50_XMZ(pretrained=True)
    backbone = res.resnet101_XMZ(pretrained=True)
    base_model = DeepLabV3Emb_512

    model = base_model(backbone, num_classes, args)

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = 2
    num_workers = args.workers

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_dataset = ReferDataset(args,
                                 split="train",
                                 image_transforms=get_transform(train=True,base_size=args.base_size, crop_size=args.crop_size,img_size=480),
                                 # image_transforms=get_transform(img_size=480),
                                 target_transforms=None,
                                )

    val_dataset = ReferDataset(args,
                               split="val",
                               image_transforms=get_transform(train=False,base_size=args.base_size, crop_size=args.crop_size,img_size=480),
                               # image_transforms=get_transform(img_size=480),
                               target_transforms=None,
                             )

    train_sampler = torch.utils.data.RandomSampler(train_dataset)  # 随机采样
    test_sampler = torch.utils.data.SequentialSampler(val_dataset)  # 顺序采样

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers,
                                               persistent_workers=True,
                                               collate_fn=utils.collate_fn_emb_berts)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             sampler=test_sampler,
                                             num_workers=num_workers,
                                             persistent_workers=True,
                                             collate_fn=utils.collate_fn_emb_berts)


    model = create_model(num_classes=num_classes, args=args)
    model.to(device)
    model_class = BertModel
    bert_model = model_class.from_pretrained('bert-base-uncased')
    bert_model.to(device)

    if args.test_only:
        confmat, iou, val_info = evaluate(model, val_loader, bert_model, device=device, num_classes=num_classes)
        print(val_info)
        return

    model_without_ddp = model
    bert_model_without_ddp = bert_model

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.parameters() if p.requires_grad]},
        # the following are the parameters of bert
        {"params": reduce(operator.concat,
                          [[p for p in bert_model_without_ddp.encoder.layer[i].parameters() if p.requires_grad] for i in
                           range(10)])},
        {"params": [p for p in bert_model_without_ddp.pooler.parameters() if p.requires_grad]}
    ]

    # optimizer = torch.optim.SGD(
    #     params_to_optimize,
    #     lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    # )

    # lr = 0.00005
    # # lr = 0.000005
    # weight_decay = 1e-2

    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=False
                                  )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=5, warmup_factor=1e-1)

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(train_loader) * args.epochs)) ** 0.9)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        bert_model.load_state_dict(checkpoint['bert_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    t_iou = 0
    for epoch in range(args.start_epoch, args.epochs):
        print('Start a new training epoch......')
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, bert_model,
                                        lr_scheduler=lr_scheduler, val_loader=val_loader, print_freq=args.print_freq,
                                        scaler=scaler)
        print('Start a new validation......')
        confmat, iou, oIOU, val_info = evaluate(model, val_loader, bert_model, device=device, num_classes=num_classes)

        print(val_info)

        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"

            f.write(train_info + val_info + "\n\n")
        if t_iou < iou:
            print('Better epoch: {}\n'.format(epoch))
            save_file = {"model": model.state_dict(),
                         'bert_model': bert_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            torch.save(save_file, "save_weights/model_{}_best.pth".format(args.model_id))
            t_iou = iou
        if epoch > 24:
            # print('Better epoch: {}\n'.format(epoch))
            save_file = {"model": model.state_dict(),
                         'bert_model': bert_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            torch.save(save_file, "save_weights/model_{}_epoch_{}.pth".format(args.model_id, epoch))
            # t_iou = iou
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)

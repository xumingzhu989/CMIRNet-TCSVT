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
import swintransformer.segmentation as swin
from src.CMIRNet_swin import CMIRNet_swin


def get_transform(train, base_size=520, crop_size=480):
    transforms = []
    transforms.append(T.RandomResize(480))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)

def create_model(num_classes, args):
    Vis_backbone_TGMM = swin.swin_backbone_TGMM(pretrained=args.pretrained, args=args)
    base_model = CMIRNet_swin

    model = base_model(Vis_backbone_TGMM, num_classes, args)

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
                                 image_transforms=get_transform(train=True),
                                 target_transforms=None,
                                 )
    val_dataset = ReferDataset(args,
                               split="val",
                               image_transforms=get_transform(train=False),
                               target_transforms=None,
                               )

    train_sampler = torch.utils.data.RandomSampler(train_dataset)  # 随机采样
    test_sampler = torch.utils.data.SequentialSampler(val_dataset)  # 顺序采样

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               pin_memory=args.pin_mem,
                                               num_workers=num_workers,
                                               collate_fn=utils.collate_fn_emb_berts)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             sampler=test_sampler,
                                             num_workers=num_workers,
                                             collate_fn=utils.collate_fn_emb_berts)

    model = create_model(num_classes=num_classes, args=args)
    model.to(device)
    # single_model = model

    model_class = BertModel
    bert_model = model_class.from_pretrained('bert-base-uncased')
    bert_model.to(device)
    # single_bert_model = bert_model
    # weights_path = "/root/project_2407/CMIRNet_swin/pretrain_weights/model_best_11_epoch_pretrain.pth"
    # weights_dict = torch.load(weights_path, map_location='cpu')
    # model.load_state_dict(weights_dict["model"])
    # bert_model.load_state_dict(weights_dict['bert_model'],  strict=False)

    if args.test_only:
        confmat, iou, val_info = evaluate(model, val_loader, bert_model, device=device, num_classes=num_classes)
        print(val_info)
        return

    model_without_ddp = model
    bert_model_without_ddp = bert_model

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.parameters() if p.requires_grad]},
        # {"params": [p for p in model_without_ddp.parameters() if p.requires_grad]},
        {"params": reduce(operator.concat,
                          [[p for p in bert_model_without_ddp.encoder.layer[i].parameters() if p.requires_grad] for i in
                           range(10)])},
        {"params": [p for p in bert_model_without_ddp.pooler.parameters() if p.requires_grad]}
    ]

    # lr = 0.01
    # momentum=0.9
    # weight_decay=1e-2
    # optimizer = torch.optim.SGD(
    #     params_to_optimize,
    #     lr=lr, momentum=momentum, weight_decay=weight_decay
    # )
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad,
                                  )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(train_loader) * args.epochs)) ** 0.9)
    # lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=10, warmup_factor=1e-1)

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
    if args.small:
        for epoch in range(args.start_epoch, args.epochs):
            mean_loss, lr, train_oiou = train_one_epoch(model, optimizer, train_loader, device, epoch, bert_model,
                                            lr_scheduler=lr_scheduler,val_loader=val_loader, print_freq=args.print_freq, scaler=scaler, args=args)

            confmat, iou,oIOU, val_info, val_loss = evaluate(model, val_loader, bert_model, device=device, num_classes=num_classes, args=args)

            print(val_info)

            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                            f"train_loss: {mean_loss:.4f}\n" \
                            f"val_loss: {val_loss:.4f}\n"\
                            f"train_iou: {train_oiou:.2f}\n"\
                            f"val_iou: {oIOU:.2f}\n"\
                            f"lr: {lr:.6f}\n"

                f.write(train_info  + "\n\n")
            if t_iou < oIOU:
                print('Better epoch: {}\n'.format(epoch))
                save_file = {"model": model.state_dict(),
                            'bert_model': bert_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args}
                if args.amp:
                    save_file["scaler"] = scaler.state_dict()
                torch.save(save_file, "save_weights/model_best_{}.pth".format(args.model_id))
                t_iou = oIOU

            if epoch<=args.epochs and epoch>=args.epochs-4:
                # print('Better epoch: {}\n'.format(epoch))
                save_file = {"model": model.state_dict(),
                            'bert_model': bert_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args}
                if args.amp:
                    save_file["scaler"] = scaler.state_dict()
                torch.save(save_file, "save_weights/model_epoch_{}_{}.pth".format(epoch, args.model_id))
                # t_iou = oIOU
    else:
        for epoch in range(args.start_epoch, args.epochs):
            mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, bert_model,
                                            lr_scheduler=lr_scheduler,val_loader=val_loader, print_freq=args.print_freq, scaler=scaler, args=args)

            confmat, iou,oIOU, val_info = evaluate(model, val_loader, bert_model, device=device, num_classes=num_classes, args=args)

            print(val_info)

            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                            f"train_loss: {mean_loss:.4f}\n" \
                            f"lr: {lr:.6f}\n"

                f.write(train_info + val_info + "\n\n")
            if t_iou < oIOU:
                print('Better epoch: {}\n'.format(epoch))
                save_file = {"model": model.state_dict(),
                            'bert_model': bert_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args}
                if args.amp:
                    save_file["scaler"] = scaler.state_dict()
                torch.save(save_file, "save_weights/model_best_{}.pth".format(args.model_id))
                t_iou = oIOU

            if epoch<=args.epochs and epoch>=args.epochs-4:
                # print('Better epoch: {}\n'.format(epoch))
                save_file = {"model": model.state_dict(),
                            'bert_model': bert_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args}
                if args.amp:
                    save_file["scaler"] = scaler.state_dict()
                torch.save(save_file, "save_weights/model_epoch_{}_{}.pth".format(epoch, args.model_id))
                # t_iou = oIOU


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


if __name__ == '__main__':
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)

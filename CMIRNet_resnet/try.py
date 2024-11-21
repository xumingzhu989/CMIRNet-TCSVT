import torch
import json
from PIL import Image
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertModel
import numpy as np
import transforms as T
from torch.backends import cudnn
import random
import os
import resnet.resnet1 as res
from src.deeplabv3_emb import DeepLabV3Emb_512
import swintransformer.segmentation as swin
from src_swinB.CMIRNet_swin import CMIRNet_swin as CMIRNet_swin_B
from src_swinL.CMIRNet_swin import CMIRNet_swin as CMIRNet_swin_L

def getSentenceAndAttention(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sentence_tokenized = tokenizer.encode(text=text, add_special_tokens=True)
    sentence_tokenized = sentence_tokenized[:20]

    padded_sent_toks = [0] * 20
    padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized
    # create a sentence token mask: 1 for real words; 0 for padded tokens
    attention_mask = [0] * 20
    attention_mask[:len(sentence_tokenized)] = [1] * len(sentence_tokenized)
    # convert lists to tensors
    padded_sent_toks = torch.tensor(padded_sent_toks).unsqueeze(0)  # (1, 20)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # (1, 20)
    return padded_sent_toks, attention_mask


def create_model(num_classes, args):
    backbone = res.resnet101_XMZ(pretrained=True)
    base_model = DeepLabV3Emb_512

    model = base_model(backbone, num_classes, args)
    return model

def create_model_swin(num_classes, args):
    Vis_backbone_TGMM = swin.swin_backbone_TGMM(pretrained=args.pretrained, args=args)
    # base_model = CMIRNet_swin_B
    base_model = CMIRNet_swin_L

    model = base_model(Vis_backbone_TGMM, num_classes, args)

    return model

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False  # 禁用benchmark，保证可复现
    # torch.backends.cudnn.benchmark = True #恢复benchmark，提升效果
    torch.backends.cudnn.deterministic = True


def main(args):
    # with open("./palette.json", "rb") as f:
    # pallette_dict = json.load(f)
    pallette = [0, 0, 0,
                128, 0, 0]
    # for v in pallette_dict.values():
    #     pallette += v
    # set_seed(1)
    # weights_path = "./save_weights/model_best_refcoco.pth"
    # weights_dict = torch.load(weights_path, map_location='cpu')

    device = torch.device("cpu")

    model = create_model_swin(num_classes=2, args=args)
    # model.load_state_dict(weights_dict["model"])

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    # bert_model.load_state_dict(weights_dict['bert_model'])

    pic_dir = "D:\有用的东西\笔记\科研\数据集\\train2014/COCO_train2014_000000581857.jpg"
    # text = "the lady with blue shirt who was back to us"
    # text = "woman in gray shirt facing camera on right"
    # text = "the green banana on the table"
    # original_img = Image.open("D:\有用的东西\笔记\科研\数据集\\train2014/COCO_train2014_000000414571.jpg")
    # text = "The little cup in the back plate"
    # pic_dir = "COCO_train2014_000000000034.jpg"
    # text = "the woman"
    # pic_dir = "/root/pythonProjects/CMPC/data/coco/images/train2014/COCO_train2014_000000002211.jpg"
    text = "the horse"
    # text = "the man in red"
    # pic_dir ="/root/xtx/dataset/train2014/COCO_train2014_000000004716.jpg"
    # text = "the man under white hat"
    # text = "the left man"
    # text = "the right man"
    # text = "the man in black"
    # text = "the logo on the wall "
    # text = "the sport ground"
    # plt.imshow(original_img)
    # pic_dir = "/root/xtx/dataset/train2014/COCO_train2014_000000121943.jpg"
    # text = "the soccer on the ground right"
    # text = "the running man"
    # text = "the soccer on the ground"
    # pic_dir = "E:\myfile\demo\demo1014.jpg"
    # text = "the computer on the left"
    # text = "the cup on the right"
    # text = "the hedgehog doll"
    # pic_dir = "D:\有用的东西\笔记\科研\数据集\\train2014/COCO_train2014_000000000839.jpg"
    # text = "the white cat in the middle of the screen"
    # pic_dir = "D:\有用的东西\笔记\科研\数据集\\train2014/COCO_train2014_000000007944.jpg"
    # text = "the knife on right "
    # pic_dir = "D:\有用的东西\笔记\科研\数据集\\train2014/COCO_train2014_000000009511.jpg"
    # text = "the apple"
    # text = "the banana"
    # pic_dir = "E:\桌面/QQ图片20221126151938.jpg"
    # text = "Woman standing inbetween the two guys"

    original_img = Image.open(pic_dir)

    data_transform = T.Compose([T.Resize(480),
                                T.ToTensor(),
                                T.Normalize(mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225))])
    img, _ = data_transform(original_img, original_img)
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        sentences, attentions = getSentenceAndAttention(text)
        last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
        sentence = last_hidden_states[:, 0, :].squeeze(1)
        embedding = last_hidden_states.permute(0, 2, 1)
        output, a, b = model(img.to(device), embedding, sentence, attentions.unsqueeze(dim=-1))
        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(prediction)
        mask.putpalette(pallette)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.text(0, 0, text, fontsize=15)
        plt.imshow(original_img)
        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.show()
        plt.close()


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()
    main(args)

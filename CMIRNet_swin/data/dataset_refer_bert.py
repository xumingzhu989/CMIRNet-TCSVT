import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random

import transformers

import h5py
from refer import REFER
from refer_small import REFER as REFER_small
from refer_merge import REFER as REFER_combined


# Dataset configuration initialization




class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 # input_size,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False,
                 ):

        self.classes = []
        # self.input_size = input_size
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        if args.combined:
            self.refer = REFER_combined("./data/", args.dataset, args.splitBy)
        elif args.small:
            self.refer = REFER_small("./data/", args.dataset, args.splitBy)
        else:
            self.refer = REFER("./data/", args.dataset, args.splitBy)

        self.max_tokens = 20

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        self.eval_mode = eval_mode

        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[PAD]'))

        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # truncation of tokens   截断
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)
       

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]
        img_name = this_img['file_name'].split('/')[-1]



        img = Image.open(os.path.join(self.refer.IMAGE_DIR, img_name)).convert("RGB")

        # img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)
        this_sent_ids = ref[0]['sent_ids']

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # involves transform from PIL to tensor and mean and std normalization
            img, target = self.image_transforms(img, annot)

        if self.eval_mode:

            embedding = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)

        else:

            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        return img, target, tensor_embeddings, attention_mask
import os  
import torch  
from torch.utils.data import Dataset  
from PIL import Image  
import numpy as np  
import transformers
import time
import datetime
from tqdm import tqdm
  
class TSVDataset(Dataset):  
    def __init__(self, tsv_file, transform=None):  
        """  
        初始化TSVDataset。  
  
        Args:  
            tsv_file_path (str): TSV文件的路径。  
            transform (callable, optional): 一个可选的转换函数/转换链，用于对样本进行预处理。  
        """  
        print("build merge dataset...")
        start_time = time.time()
        self.transform = transform  
        self.data = []  
        self.max_tokens = 20
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        tsv_file_path = os.path.join('/root/xtx/project_2404/CMIRNet_A40_48G_SwinB_3/data/pretrain',tsv_file+'.tsv')

        # 读取TSV文件  
        with open(tsv_file_path, mode='r', encoding='utf-8') as file:  
            for i,line in enumerate(file):  
                parts = line.strip().split('\t')  # 使用制表符分割每行  
                if len(parts) >= 4:  # 确保有足够的数据  
                    img_path = parts[3]  # 假设图片路径是第四个字段  
                    coordinates = parts[2].split(',')  # 分割坐标字符串  
            # 现在coordinates是一个列表，包含了坐标值，如果需要转换为浮点数列表，可以这样做：  
                    coordinates_floats = np.array([float(coord) for coord in coordinates])
                    coordinates_floats = torch.tensor(coordinates_floats).unsqueeze(0).unsqueeze(0)
                    language = parts[1]
                    attention_mask = [0] * self.max_tokens
                    padded_input_ids = [0] * self.max_tokens

                    input_ids = self.tokenizer.encode(text=language, add_special_tokens=True)

                    # truncation of tokens   截断
                    input_ids = input_ids[:self.max_tokens]

                    padded_input_ids[:len(input_ids)] = input_ids
                    attention_mask[:len(input_ids)] = [1]*len(input_ids)

                    sentences_for_ref = torch.tensor(padded_input_ids).unsqueeze(0)
                    attentions_for_ref = torch.tensor(attention_mask).unsqueeze(0)
 
                    self.data.append((img_path,sentences_for_ref,attentions_for_ref, coordinates_floats))  # 将图片路径和坐标（或其他需要的数据）作为一个元组添加  
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('build success!')
        print(f"total time:{total_time_str}")
    def __len__(self):  
        """  
        返回数据集中的样本总数。  
        """  
        return len(self.data)  
  
    def __getitem__(self, idx):  
        """  
        根据索引获取样本。  
  
        Args:  
            idx (int): 样本的索引。  
  
        Returns:  
            tuple: 一个包含样本数据和标签（如果有的话）的元组。  
                在这个例子中，我们返回图片（作为PIL Image或转换后的Tensor）和坐标（作为字符串或转换后的格式）。  
        """  
        img_path,language,attention, coords = self.data[idx]  
          
        # 加载图片  
        image = Image.open(img_path).convert('RGB')  
        # width, height = image.size 
        # matrix = np.zeros((height, width), dtype=np.uint8) 
        # coord = np.indices(matrix.shape)  
        # rect_mask = (coord[0] >= coords[1]) & (coord[0] < coords[3]) & (coord[1] >= coords[0]) & (coord[1] < coords[2])  
  
# 将矩形框内的值设置为1  
        # matrix[rect_mask] = 1
        # coords = Image.fromarray(matrix.astype(np.uint8), mode="P") 

  
        # 如果存在转换函数，则应用它  
        if self.transform:  
            image, coords = self.transform(image, coords)  
  
        # 返回图片和坐标（或转换后的格式）  
        return image, coords,language,attention  # 或者返回 image, coords（如果coords已经被转换）  
  
# 使用示例  
# 假设你有一个名为'your_dataset.tsv'的TSV文件和一个转换函数transform  
# transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])  
# dataset = TSVDataset('your_dataset.tsv', transform=transform)  
# 你可以使用dataset来创建DataLoader，以便在训练循环中迭代数据
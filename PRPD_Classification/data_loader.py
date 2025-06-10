import os
import random
import numpy as np
import torch 
from torch.utils import data 
from torchvision import transforms as T 
from PIL import Image 
import pandas as pd


class PRPD_Classification(data.Dataset): 
	def __init__(self, Data_set, mode='train'): 
		self.image_paths = Data_set
		self.mode = mode 
		self.set_seed(42) 
		print("image count in {} path :{}".format(self.mode,len(self.image_paths))) 

	def __getitem__(self, index): 
		data_path = self.image_paths[index] 
		image = Image.open(data_path).convert('RGB') 

		transform = T.Compose([
			T.ToTensor(),
			T.Resize((224, 224)), # 이미지 크기를 224x224로 조정
			T.Normalize(mean=[0.485, 0.456, 0.406],
						std=[0.229, 0.224, 0.225])
		])

		image = transform(image)
		label_path = data_path.replace('원천데이터','라벨링데이터').replace('.png','.json')
		label = pd.read_json(label_path, encoding='utf-8-sig')
		label = int(label['label']['PD_type'])
		label = torch.tensor(label)
		return image, label, self.image_paths[index]


	def set_seed(self,seed):
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		np.random.seed(seed)
		random.seed(seed)
		os.environ['PYTHONHASHSEED'] = str(seed)
  

	def __len__(self):
		return len(self.image_paths)


def get_loader( Data_set, batch_size, num_workers=2, mode='train' ): 
	dataset = PRPD_Classification(Data_set = Data_set, mode=mode)

	data_loader = data.DataLoader( 
									dataset=dataset, 
									batch_size=batch_size, 
									shuffle=True,
									num_workers=num_workers
									)
	return data_loader
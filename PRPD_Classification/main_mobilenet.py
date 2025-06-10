import os
import torch
import random
import argparse
import numpy as np
from torch.backends import cudnn
from make_dataset import make_dataset

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(config):
    set_seed(42)
    if config.mode == 'train':

        from data_loader import get_loader
        train_dataset, valid_dataset, test_dataset = make_dataset(config.raw_data_path, config.dataset_ratio, config.dataset_type)
        train_loader = get_loader(
                                Data_set=train_dataset,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                mode='train'
                                )
        valid_loader = get_loader( 
                                Data_set=valid_dataset, 
                                batch_size=config.batch_size, 
                                num_workers=config.num_workers, 
                                mode='valid'
                                )
        test_loader = get_loader(
                                Data_set=test_dataset, 
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                mode='test'
                                )
        from train import Train
            
        train = Train(config, train_loader, valid_loader, test_loader)
        train.train() 

if __name__ == '__main__':  

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_ch', type=int, default=3) # img_channel
    parser.add_argument('--output_ch', type=int, default=5) # output_channel
    parser.add_argument('--num_epochs', type=int, default=100) # Epoch 
    parser.add_argument('--num_epochs_decay', type=int, default=70) # Learning Rate 설정을 위한 num_epochs_decay
    parser.add_argument('--num_classes', type=int, default=5) # 분류를 위한 클래스
    parser.add_argument('--batch_size', type=int, default=128) # batch_size 
    parser.add_argument('--num_workers', type=int, default=4) # num_workers 
    parser.add_argument('--lr', type=float, default=0.0001) # Learning Rate
    parser.add_argument('--beta1', type=float, default=0.5) # beta1    
    parser.add_argument('--beta2', type=float, default=0.999) # beta2  
    parser.add_argument('--patience', type=int, default=5) # Early Stop Patience 
    parser.add_argument('--model_type', type=str, default='Mobilenet', help='Resnet/Efficientnet/Mobilenet/Vit/Convnext') # 모델 유형 입력(Resnet/Efficientnet/Mobilenet/Vit/Convnext)
    parser.add_argument('--model_path', type=str, default='./PRPD_Classification/Mobilenet_Result') # 학습 결과 저장 경로
    parser.add_argument('--load_model_path', type=str, default='') 
    parser.add_argument('--dataset_ratio',type=float, default=1)
    parser.add_argument('--raw_data_path', type=str, default='./원천데이터/') # 원본 데이터셋 경로
    parser.add_argument('--result_path', type=str, default='./PRPD_Classification/Mobilenet_Result/') # 모델 학습 결과 저장 경로
    parser.add_argument('--mode', type=str, default='train', help='train')
    parser.add_argument('--dataset_type', type=str, default='전체', help='전체/고체/액체/기체')  # 모델 학습에 사용할 데이터셋 절연체 종류 입력(전체/고체/액체/기체)
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0/cuda:1')
    config = parser.parse_args()
    main(config)
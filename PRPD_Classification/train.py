import os
import torch
from torch import optim
import torch.nn.functional as F
from tqdm.auto import tqdm
import csv
import torch.nn as nn
import torchvision.transforms as T
from module import *
import datetime
import matplotlib.pyplot as plt

train_loss_list = [] 
valid_loss_list = []
test_loss_list = []
train_acc = []
valid_acc = []
test_acc = []

class Train(object): 
	def __init__(self, config, train_loader, valid_loader, test_loader):
		self.train_loader = train_loader; self.valid_loader = valid_loader; self.test_loader = test_loader
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2
		self.patience = config.patience
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		self.start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
		self.model_path = config.model_path
		self.model_type = config.model_type
		self.result_path = config.result_path + self.model_type +'_' + self.start_time + '/'
		os.makedirs(self.model_path, exist_ok=True)
		os.makedirs(self.result_path, exist_ok=True)
		self.load_model_path = config.load_model_path
		self.num_classes = config.num_classes
		self.device = config.device
		self.model = build_model(config, mode = 'train', result_path = self.result_path, device = self.device)
		self.optimizer = optim.Adam(list(self.model.parameters()), self.lr, [self.beta1, self.beta2])
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = nn.CrossEntropyLoss()

	def train(self):
		f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
		t = open(os.path.join(self.result_path,'test.csv'), 'a', encoding='utf-8', newline='')
		wr = csv.writer(f) 
		tr = csv.writer(t) 
		wr.writerow(['model_type','Loss','Accuracy','Sensitivity','Precision','F1_score','Learning_rate','Best_epoch','Num_epochs','Num_epochs_decay'])
		tr.writerow((['model_type','Loss','Accuracy','Sensitivity','Precision','F1_score','Best_epoch']))
		if self.load_model_path != '':
			self.model.load_state_dict(torch.load(self.load_model_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,self.load_model_path))
		f.close()
		t.close()
		lr = self.lr
		early_stop = 0
		best_model_score = 100.
		
		for epoch in range(self.num_epochs):
			print('\n{}'.format('-'*100))
			self.model.train(True)
			epoch_loss = 0; best_epoch = 0
			acc, SE, PC, F1  =  0.,0.,0.,0.
			all_predictions = [] 
			all_labels = [] 
			for i, (images, label, _) in enumerate(tqdm(self.train_loader,desc="[Training]")):
				f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
				t = open(os.path.join(self.result_path,'test.csv'), 'a', encoding='utf-8', newline='')
				wr = csv.writer(f)
				tr = csv.writer(t)
				images = images.to(self.device)
				label = label.to(self.device)
				result = self.model(images)
				loss = self.criterion(result,label)
				epoch_loss += loss.item()
				self.model.zero_grad()
				loss.backward()
				self.optimizer.step()
				all_predictions.extend(F.softmax(result, dim=1).argmax(dim=1).detach().cpu().numpy())
				all_labels.extend(label.detach().cpu().numpy())
				self.model.zero_grad()
    
			length=len(self.train_loader)
			acc,  SE, PC, F1  = evaluate(acc, SE, PC, F1, all_predictions, all_labels)
			epoch_loss = epoch_loss/length

			print('[Training] Epoch: %d/%d, Loss: %.8f, Acc: %.4f, SE: %.4f, PC: %.4f, F1: %.4f' % (epoch+1, self.num_epochs, epoch_loss, acc,SE,PC,F1)) #  Epoch 당 성능(Loss, 정확도, 민감도, 정밀도, F1-Score) 터미널 출력
			train_loss_list.append(epoch_loss)
			train_acc.append(acc)

			if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
				lr -= (self.lr / float(self.num_epochs_decay))
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = self.lr
				print ('Decay learning rate to lr: {}.'.format(lr))
			
			
			self.model.train(False)
			self.model.eval()

			acc, SE, PC, F1  =  0.,0.,0.,0.
			with torch.no_grad(): 
				all_predictions = [] 
				all_labels = [] 
				for i, (images, label, raw_name) in enumerate(tqdm(self.valid_loader,desc="[Validation]")): 
					images = images.to(self.device)
					label = label.to(self.device)
					result = self.model(images)
					valid_loss = self.criterion(result,label)
					valid_loss += valid_loss.item()
					all_predictions.extend(F.softmax(result, dim=1).argmax(dim=1).detach().cpu().numpy())
					all_labels.extend(label.detach().cpu().numpy())
					
				
				length=len(self.valid_loader) # Epoch 당 성능 갱신을 위한 Batch Size 반환
				acc,  SE, PC, F1  = evaluate(acc, SE, PC, F1, all_predictions, all_labels) # 정확도, 민감도, 정밀도, F1-Score 연산 및 갱신
				valid_loss = valid_loss.item()/length # Epoch 당 Loss 갱신
				model_score = valid_loss # Early Stop 적용을 위한 기준 갱신
				print('[Validation] Loss: %.8f, Acc: %.4f, SE: %.4f, PC: %.4f, F1: %.4f'%(valid_loss,acc,SE,PC,F1))  #  Epoch 당 성능 (Loss, 정확도, 민감도, 정밀도, F1-Score) 터미널 출력
				valid_loss_list.append(valid_loss) # valid_loss를 valid_loss_list에 추가
				valid_acc.append(acc) # valid_acc를 valid_acc 리스트에 추가

			if model_score < best_model_score: # model_score가 best_model_score보다 작을때 작동
				test_flag = 1 # test를 위한 변수 선언
				best_model_score = model_score # best_model_score 갱신 
				best_epoch = epoch+1 # best_epoch을 현재 epoch으로 갱신
				early_stop = 0 # early_stop 카운트 초기화 
				best_model = self.model.state_dict() # best_model을 현재 모델의 가중치로 갱신 
				model_path = os.path.join(self.model_path, '%s_%s_%d.pkl' %(self.model_type,self.start_time,epoch+1)) # 갱신된 best_model을 저장할 경로를 model_path에 저장
				torch.save(best_model,model_path) # best_model 저장
				visualize_weight_distribution(self.model, os.path.join(self.result_path,'%s_valid_%d_Weights.png'%(self.model_type,epoch+1))) # 갱신된 best_model의 가중치 분포 시각화
				
			else: # model_score가 best_model_score보다 클때 작동
				early_stop += 1 # early_stop을 위한 카운트 증가
				
			wr.writerow([self.model_type,valid_loss,acc,SE,PC,F1,self.lr,best_epoch,self.num_epochs,self.num_epochs_decay]) # 모델 학습/검증 결과 저장
			#===================================== Test ====================================#
			self.model.train(False) # 모델 학습 모드 종료
			self.model.eval() # 모델 평가 모드 시작

			acc, SE, PC, F1  =  0.,0.,0.,0. # 정확도, 민감도, 정밀도, F1-Score 갱신을 위한 변수 선언
			with torch.no_grad(): # 모델 검증을 위한 기울기 갱신 기능 종료
				all_predictions = []
				all_labels = []
				for i, (images, label, raw_name) in enumerate(tqdm(self.test_loader,desc="[Test]")): # Test Loop

					images = images.to(self.device) # Test Dataset의 이미지를 Batch_Size 만큼 GPU 업로드
					label = label.to(self.device) # Test Dataset의 이미지에 매칭되는 정답정보(정상, 노이즈, 표면 방전, 코로나 방전, 보이드 방전)를 Batch_Size 만큼 GPU 업로드
					result = self.model(images) # 입력 이미지에 대한 결과 result 변수에 저장
					test_loss = self.criterion(result,label) # 결과와 정답에 대한 CrossEntropyLoss 연산 및 저장
					test_loss += test_loss.item() # CrossEntropyLoss 갱신
					all_predictions.extend(F.softmax(result, dim=1).argmax(dim=1).detach().cpu().numpy()) # 혼동 행렬 추출을 위한 예측값 계산 및 all_predictions에 추가 
					all_labels.extend(label.detach().cpu().numpy()) # 혼동 행렬 추출을 위한 모든 정답값 all_labels 리스트에 추가
     
				length=len(self.test_loader) # 성능 계산을 위한 Batch Size 반환
				acc,  SE, PC, F1  = evaluate(acc, SE, PC, F1, all_predictions, all_labels) # 정확도, 민감도, 정밀도, F1-Score 연산 및 갱신
				test_loss = test_loss.item()/length # Test Loss 계산
				print('[Test] Loss: %.8f, Acc: %.4f, SE: %.4f, PC: %.4f, F1: %.4f'%(test_loss,acc,SE,PC,F1)+'\n{}'.format('-'*100)) # Test 성능(Loss, 정확도, 민감도, 정밀도, F1-Score) 터미널 출력
				test_loss_list.append(test_loss) # Test Loss를 test_loss_list에 추가
				test_acc.append(acc) # Test Accuracy를 test_acc 리스트 추가
				
    
			if test_flag == 1: # test_flag 변수가 1일 때 작동
				test_flag = 0 # test_flag 초기화
			tr.writerow([self.model_type,test_loss,acc,SE,PC,F1,self.lr,best_epoch]) # 모델 TEST 결과 저장을 위한 CSV파일 헤더 쓰기
			save_confusion_matrix(all_labels,all_predictions,self.model_type,self.result_path+str(epoch+1)) # TEST 혼동 행렬 저장
			if early_stop >= self.patience: # early_stop이 patience 보다 크거나 같을 때 작동
				pos = len(valid_loss_list)-self.patience # x축 위치 지정을 위한 valid_loss_list의 길이에서 patience만큼 뺀 값을 길이로 지정
				print(f'Early Stopping ({early_stop} / {self.patience})') # early_stop으로 인한 학습 종료 설정
				epochs = range(1,len(train_loss_list)+1) # 1에폭 부터 Plot을 위해 Epoch의 범위를 1부터 train_loss_list길이에 1을 더해 지정
				plt.plot(epochs, train_loss_list, 'r', label='Training loss', alpha = 0.6) # train_loss_list Plot
				plt.plot(epochs, valid_loss_list, 'g', label='Validation loss', alpha = 0.6) # valid_loss_list Plot
				plt.plot(epochs, test_loss_list, 'b', label='Test loss', alpha = 0.6) # test_loss_list Plot
				plt.axvline(x=pos, c='y') # x축 pos 위치에 노란색 수직선 추가
				plt.title('Train/Validation/Test loss') # Train/Validation/Test loss라는  title 지정
				plt.xlabel('Epochs') # x축 이름 지정
				plt.ylabel('Loss') # y축 이름 지정
				plt.legend() # 범례 지정
				plt.savefig(os.path.join(self.result_path,'loss.png')) # Plot 저장
				plt.close() # Plot 종료

				plt.plot(epochs, train_acc, 'r', label='Training accuracy', alpha = 0.6) 
				plt.plot(epochs, valid_acc, 'g', label='Validation accuracy', alpha = 0.6) 
				plt.plot(epochs, test_acc, 'b', label='Test accuracy', alpha = 0.6) 
				plt.axvline(x=pos, c='y') 
				plt.title('Train/Validation/Test Accuracy') 
				plt.xlabel('Epochs') 
				plt.ylabel('Accuracy') 
				plt.legend() 
				plt.savefig(os.path.join(self.result_path,'accuracy.png')) 
				plt.close() 
				break 
			else: 
				print('Best %s model score : %.8f'%(self.model_type,best_model_score)) 
				print(f'Early Stopping ({early_stop} / {self.patience})'+'\n{}'.format('-'*100)) 
			f.close()
			t.close()


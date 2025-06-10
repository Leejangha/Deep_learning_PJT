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
					
				
				length=len(self.valid_loader)
				acc,  SE, PC, F1  = evaluate(acc, SE, PC, F1, all_predictions, all_labels)
				valid_loss = valid_loss.item()/length 
				model_score = valid_loss
				print('[Validation] Loss: %.8f, Acc: %.4f, SE: %.4f, PC: %.4f, F1: %.4f'%(valid_loss,acc,SE,PC,F1))
				valid_loss_list.append(valid_loss)
				valid_acc.append(acc)

			if model_score < best_model_score: 
				test_flag = 1
				best_model_score = model_score
				best_epoch = epoch+1
				early_stop = 0
				best_model = self.model.state_dict()
				model_path = os.path.join(self.model_path, '%s_%s_%d.pkl' %(self.model_type,self.start_time,epoch+1))
				torch.save(best_model,model_path)
				visualize_weight_distribution(self.model, os.path.join(self.result_path,'%s_valid_%d_Weights.png'%(self.model_type,epoch+1)))
				
			else:
				early_stop += 1
				
			wr.writerow([self.model_type,valid_loss,acc,SE,PC,F1,self.lr,best_epoch,self.num_epochs,self.num_epochs_decay])

			self.model.train(False)
			self.model.eval()

			acc, SE, PC, F1  =  0.,0.,0.,0.
			with torch.no_grad():
				all_predictions = []
				all_labels = []
				for i, (images, label, raw_name) in enumerate(tqdm(self.test_loader,desc="[Test]")):

					images = images.to(self.device)
					label = label.to(self.device)
					result = self.model(images)
					test_loss = self.criterion(result,label)
					test_loss += test_loss.item()
					all_predictions.extend(F.softmax(result, dim=1).argmax(dim=1).detach().cpu().numpy())
					all_labels.extend(label.detach().cpu().numpy())
     
				length=len(self.test_loader)
				acc,  SE, PC, F1  = evaluate(acc, SE, PC, F1, all_predictions, all_labels)
				test_loss = test_loss.item()/length
				print('[Test] Loss: %.8f, Acc: %.4f, SE: %.4f, PC: %.4f, F1: %.4f'%(test_loss,acc,SE,PC,F1)+'\n{}'.format('-'*100))
				test_loss_list.append(test_loss)
				test_acc.append(acc)
				
    
			if test_flag == 1:
				test_flag = 0
			tr.writerow([self.model_type,test_loss,acc,SE,PC,F1,self.lr,best_epoch])
			save_confusion_matrix(all_labels,all_predictions,self.model_type,self.result_path+str(epoch+1))
			if early_stop >= self.patience:
				pos = len(valid_loss_list)-self.patience
				print(f'Early Stopping ({early_stop} / {self.patience})')
				epochs = range(1,len(train_loss_list)+1)
				plt.plot(epochs, train_loss_list, 'r', label='Training loss', alpha = 0.6)
				plt.plot(epochs, valid_loss_list, 'g', label='Validation loss', alpha = 0.6)
				plt.plot(epochs, test_loss_list, 'b', label='Test loss', alpha = 0.6)
				plt.axvline(x=pos, c='y')
				plt.title('Train/Validation/Test loss')
				plt.xlabel('Epochs')
				plt.ylabel('Loss')
				plt.legend()
				plt.savefig(os.path.join(self.result_path,'loss.png'))
				plt.close()

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


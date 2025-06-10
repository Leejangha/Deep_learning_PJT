import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from module import *
import datetime
from gradcam import generate_and_save_gradcam
import os


test_loss_list = [] 
test_acc = [] 

class Start(object): 
	def __init__(self, config, test_loader):
		self.test_loader = test_loader
		self.batch_size = config.batch_size # 모델 학습을 위한 배치사이즈 설정
		self.start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # 실험 결과 기록을 위한 학습 시작 시간 저장을 위한 변수 선언
		self.model_type = config.model_type # Resnet/Efficientnet과 같이 모델 유형 설정을 위한 변수 선언
		self.load_model_path = config.load_model_path
		self.num_classes = config.num_classes # 클래스 개수 선언 (정상, 노이즈, 표면 방전, 코로나 방전, 보이드 방전)
		self.device = config.device
		self.model = build_model(config, mode = 'test', result_path = '', device = self.device)
		self.img_ch = config.img_ch # img_channel 선언
		self.output_ch = config.output_ch # output_channel 선언
		

	def start(self): # 모델 test결과에 대한 혼동행렬 저장 함수
		if self.load_model_path != '': 
			self.model.load_state_dict(torch.load(self.load_model_path,map_location='cuda:0')) # 사전학습된 모델 경로를 통한 모델 가중치 로드 및 모델에 적용
			print('%s is Successfully Loaded from %s'%(self.model_type,self.load_model_path))

		self.model.train(False)
		self.model.eval()

		with torch.no_grad():
			all_predictions = [] 
			all_labels = [] 
			raw_names = [] 
			all_probability = []
			for i, (images, label, raw_name) in enumerate(tqdm(self.test_loader,desc="[Test]")):

				images = images.to(self.device)
				label = label.to(self.device)
				result = self.model(images)

				preds = F.softmax(result, dim=1).argmax(dim=1)

				all_predictions.extend(F.softmax(result, dim=1).argmax(dim=1).detach().cpu().numpy())
				all_labels.extend(label.detach().cpu().numpy())
				raw_names.extend(raw_name)
				all_probability.extend(F.softmax(result,dim=1).cpu().numpy())

		# Grad-CAM
		for idx in range(images.size(0))[:10]:
			img_tensor = images[idx:idx+1]
			cls_idx    = preds[idx].item()

			# 모델 종류에 맞게 훅을 걸 레이어 선택
			target_layer = self.model.backbone.features[-1]

			# 파일명에 인덱스를 붙여서 구분
			fname = os.path.splitext(os.path.basename(raw_name[idx]))[0]
			save_prefix = os.path.join(
				'./results/grad-cam/', f"gradcam_{fname}")

			# CAM 생성 & 저장
			generate_and_save_gradcam(
				self.model,
				img_tensor,
				target_layer,
				cls_idx,
				save_prefix
				)
		with open('./PRPD_Classification/{}_혼동행렬.csv'.format(self.model_type), 'w', encoding='euc-kr') as f:
			f.write('Class ID,정상/정상,정상/노이즈,정상/표면방전,정상/코로나방전,정상/보이드방전,노이즈/정상,노이즈/노이즈,노이즈/표면방전,노이즈/코로나방전,노이즈/보이드방전,표면방전/정상,표면방전/노이즈,표면방전/표면방전,표면방전/코로나방전,표면방전/보이드방전,코로나방전/정상,코로나방전/노이즈,코로나방전/표면방전,코로나방전/코로나방전,코로나방전/보이드방전,보이드방전/정상,보이드방전/노이즈,보이드방전/표면방전,보이드방전/코로나방전,보이드방전/보이드방전\n')
			for i in tqdm(range(len(all_labels))):
				cm = confusion_matrix([all_labels[i]], [all_predictions[i]], labels=[0,1,2,3,4])
				f.write(raw_names[i].split('/')[-1][0:-4]+',') 
				f.write(','.join(map(str, cm.flatten()))+'\n') 
		with open('./PRPD_Classification/{}_모델예측값.csv'.format(self.model_type), 'w', encoding='euc-kr') as f: 
			f.write('모델예측값,정상,노이즈,표면방전,코로나방전,보이드방전\n') 
			for i in tqdm(range(len(all_probability))): 
				f.write(raw_names[i].split('/')[-1][0:-4]+',') 
				f.write(','.join(map(str, list(all_probability[i])))+'\n') 
		acc,  SE, PC, F1 = 0,0,0,0
		acc,  SE, PC, F1  = evaluate(acc, SE, PC, F1, all_predictions, all_labels) 
		print('[Test] Acc: %.4f, SE: %.4f, PC: %.4f, F1: %.4f'%(acc,SE,PC,F1)+'\n{}'.format('-'*100))
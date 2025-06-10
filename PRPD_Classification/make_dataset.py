import os
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

def extract_time_from_filename(file_name): # 파일의 이름에서 시간 정보만 추출
    match = re.search(r'(\d{6}_\d{6})', file_name)
    if match:
        return match.group(1)
    return "" 

def path_maker(base_path, dataset_type): # 원천데이터 경로 반환 함수 
    
    dataset_type_dict = {'전체':['고체', '액체', '기체'], '고체':['고체'], '액체':['액체'], '기체':['기체']} # 데이터셋 생성을 위한 딕셔너리 
    pd_type_check_list = ['정상','노이즈','표면 방전','표면방전','보이드 방전','보이드방전','코로나 방전','코로나방전']
    insulator_check_list = dataset_type_dict[dataset_type]
    equipment_check_list = ['ACSR-OC','CNCV-W','TFR-CV','7.2kV배전반','22.9kV배전반','25.8kVGIS','단상유입변압기','전력용유입변압기','계기용변압기']
    all_folders = [] 

    pd_type_list = os.listdir(base_path)
    for pd_type in pd_type_list:
        if pd_type in pd_type_check_list:
            pd_type_path = base_path + pd_type + '/' 
            insulator_list = os.listdir(pd_type_path) 
            for insulator in insulator_list: 
                if insulator in insulator_check_list: 
                    insulator_path = pd_type_path + insulator + '/' 
                    equipment_list = os.listdir(insulator_path) 
                    for equipment in equipment_list: 
                        if equipment in equipment_check_list: 
                            equipment_path = insulator_path + equipment + '/' 
                            all_folders.append(equipment_path) 
    return all_folders 

def make_dataset(current_path, ratio=1, dataset_type = '전체'): 

    all_folders = path_maker(current_path, dataset_type) 
    train_dataset = [] 
    valid_dataset = [] 
    test_dataset = []  

    for path in all_folders: 
        
        files = [path + f for f in os.listdir(path) if f.lower().endswith('.png')] 
        
        files.sort(key=lambda x: extract_time_from_filename(os.path.basename(x))) 
        
        total_files = len(files)
        train_files = files[:int(total_files * 0.8 * ratio)] # 전체 데이터 중 0.8비율 만큼 Train 데이터셋 생성
        valid_files = files[int(total_files * 0.8 * ratio):int(total_files * 0.9 * ratio)] #  Train 데이터셋을 제외한 0.2 비율 만큼의 데이터 중 0.1 만큼의 데이터를 통해  Validation 데이터셋 생성
        test_files  = files[int(total_files * 0.9 * ratio):int(total_files * ratio)] # 마지막 0.1 만큼의 데이터를 통해 Test 데이터셋 생성
            
        train_dataset.extend(train_files)
        valid_dataset.extend(valid_files)
        test_dataset.extend(test_files)

    return train_dataset, valid_dataset, test_dataset
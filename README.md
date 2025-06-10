![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,2,2,3&height=250&section=header&text=DeepLearning&animation=fadeIn&fontColor=d6ace6&fontSize=90)

> ## :alarm_clock: 개발기간

2025.05.27(화) ~ 2025.06.10(화)

> ## 🤝 팀원

| **김동호** | **이장하** | **홍성문** |
| ---------- | ---------- | ---------- |

> ## :page_facing_up: 주제

부분방전 이미지 데이터셋 기반 화재 사고 분류 모델 개발

> ## :white_check_mark: 데이터셋
* 데이터 소스 : 산업 설비 전기 화재 사고 예방 부분방전 데이터

  https://www.aihub.or.kr/aihubdata/data/view.do?searchKeyword=%EB%B0%A9%EC%A0%84&aihubDataSe=data&dataSetSn=71682

* 데이터 구축년도/데이터 구축량 : 2023년/이미지 300,000건, 시계열 300,000건

> ## :page_with_curl: 실행 방법

1. Repository를 Clone

```
git clone https://github.com/Leejangha/Deep_learning_PJT.git
```

2. 가상환경 설치

```
conda create -n pjt python==3.8
```

3. requirements.txt 설치

```
pip install -r requirements.txt
```

4. 데이터 다운로드 후 파일 경로(원천데이터, 라벨링데이터)에 맞게 저장
5. 원하는 모델 실행

```
python3 ./PRPD_Classification/main_mobilenet.py   —raw_data_path ./원천데이터/
```




> ## :books: 기술스택

<div align=center><h1>📚 STACKS</h1></div>
<div align=center>
    <br>
    <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
    <img src="https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white">
</div>
> ## :link: API

https://www.aihub.or.kr/devsport/apishell/list.do?currMenu=403&topMenu=100


> ## :file_folder: Project Structure (프로젝트 구조)

```plaintext
PRPD_Classification/	 # 메인 코드 파일
라벨링데이터/				 # 라벨링 데이터
원천데이터/				  # 이미지 및 시계열 데이터
results/				 # 결과 저장 경로
├── grad-cam/			 # 이미지 파일
.gitignore               # Git 무시 파일 목록
README.md                # 프로젝트 개요 및 사용법
requirements.txt		 # 정확한 종속성 버전이 기록된 파일로, 일관된 빌드를 보장
```

<br/>
<br/>

> ## :soccer: 실험 결과

### MobileNet Grad-cam

<p align="center">
    <img src="https://github.com/user-attachments/assets/e02a8256-c32b-47b4-90e1-41ca77d5360e">
</p>

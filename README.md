# Cat Re-ID Prototype

Hello Street Cat 프로젝트의 고양이 개체 식별(Re-Identification) 자동화를 위한 프로토타입 프로젝트입니다. MegaDescriptor(동물 전용 foundation model)를 기반으로 고양이의 특징을 추출하고 KNN(코사인 유사도)을 통해 개체를 판별합니다.

## 📁 파일 트리 및 아키텍처

```text
cat_reid/
├── config/
│   └── settings.py          # 임계값, 모델 버전(Swin-L), 경로 등 중앙 설정
├── models/
│   └── extractor.py         # FeatureExtractor: MegaDescriptor를 이용한 특징 벡터(Embedding) 추출
├── db/
│   └── embedding_store.py   # EmbeddingStore: 추출된 벡터와 레이블을 numpy/pickle 기반으로 저장/로드
├── inference/
│   └── knn_classifier.py    # KnnClassifier: 코사인 유사도 기반 KNN 분류 (Threshold로 Unknown 판별)
├── preprocessing/
│   └── transform.py         # 이미지 전처리 파이프라인 (Resize, Normalize)
├── input/
│   ├── camera.py            # 실시간 웹캠 입력 및 결과 오버레이 처리
│   ├── file.py              # 단일 이미지 파일 추론 파이프라인
│   └── register.py          # 이미지에서 특징 추출 후 DB에 등록
├── visualization/
│   └── renderer.py          # OpenCV를 이용한 Bounding box, 이름, 유사도 화면 출력
├── evaluation/
│   └── metrics.py           # Evaluator: Top-k 정확도 및 추론 속도(Latency) 측정
├── data/
│   └── loader.py            # CatDatasetLoader: 데이터 분할(Gallery/Query) 및 PyTorch DataLoader 제공
└── training/
    └── trainer.py           # CatReIdModel: 1.5단계를 위한 Backbone 고정 및 Projection 레이어 학습 스크립트

main.py                      # 프로젝트 통합 실행 엔트리포인트 (CLI 인터페이스)
```

## 🚀 환경 설정 및 설치

```bash
# 가상 환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 요구사항 설치
pip install torch torchvision timm scikit-learn numpy opencv-python pillow tqdm
```

> **Note**: Hugging Face Hub에서 모델을 다운로드할 때 `Warning: You are sending unauthenticated requests...` 메시지가 뜰 수 있으나, 정상 작동에는 영향을 주지 않습니다.

## 🛠️ 실행 방법 (`main.py`)

`main.py`는 `--mode` 인자를 통해 다양한 기능을 지원합니다.

### 1. 고양이 등록 (DB 구축)
폴더 구조(`dataset/개체명/*.jpg`)를 순회하며 특징을 추출해 DB(`db/embeddings.npy`)를 생성합니다.
```bash
python main.py --mode register --path ./heellostreetcat-individuals
```
*(단일 이미지 등록 시: `--path 이미지경로 --label 고양이이름`)*

### 2. 단일 이미지 테스트
등록된 DB를 바탕으로 입력된 이미지의 개체를 판별합니다.
```bash
python main.py --mode infer --path ./test_image.jpg
```

### 3. 실시간 카메라 테스트
웹캠을 연결하여 실시간으로 고양이 얼굴을 비추면 화면에 이름과 유사도를 표시합니다.
```bash
python main.py --mode camera --source 0
```

### 4. 성능 평가 (Evaluate)
전체 데이터셋을 Gallery(등록용)와 Query(테스트용)로 자동 분할(`--test_size 0.5`)하여 정확도(Accuracy)와 속도(FPS)를 평가합니다.
```bash
python main.py --mode evaluate --path ./heellostreetcat-individuals
```

### 5. 1.5단계 모델 학습 (Fine-tuning)
MegaDescriptor의 가중치는 고정하고, 프로젝트 전용 Projection Layer를 학습시킵니다.
```bash
python main.py --mode train --path ./heellostreetcat-individuals --epochs 10
```
*(Colab 환경에서의 실행을 권장합니다.)*

## ⚙️ 주요 설정 옵션 (`config/settings.py`)
- `THRESHOLD = 0.7`: 이 값보다 코사인 유사도가 낮으면 등록되지 않은 고양이(`Unknown`)로 판별합니다.
- `MODEL_NAME`: 현재 `hf-hub:BVRA/MegaDescriptor-L-384` (Swin-Large)를 사용 중입니다. 엣지 디바이스로 변경 시 Tiny 모델로 수정할 수 있습니다.

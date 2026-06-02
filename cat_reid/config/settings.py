import os

# 모델 설정
MODEL_NAME = "hf-hub:BVRA/MegaDescriptor-L-384"
INPUT_SIZE = (384, 384)

# Re-ID 설정
THRESHOLD = 0.7  # 동일 개체 판별 임계값
K = 1            # KNN의 K 값

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "heellostreetcat-individuals")
DB_PATH = os.path.join(BASE_DIR, "db", "embeddings.npy")
LABEL_PATH = os.path.join(BASE_DIR, "db", "labels.pkl")

# 로깅 설정
LOG_LEVEL = "INFO"

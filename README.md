  1. 주요 구현 내용
   - config/settings.py: 모델명, 임계값(THRESHOLD=0.7), 경로 등 설정 관리
   - models/extractor.py: MegaDescriptor(Swin-L)를 이용한 특징 추출 (L2 정규화 포함)
   - db/embedding_store.py: 특징 벡터 및 레이블 저장/로드 (numpy, pickle 사용)
   - inference/knn_classifier.py: 코사인 유사도 기반 KNN 분류 및 Unknown 판별
   - input/: 이미지 등록(register.py), 파일 추론(file.py), 실시간 카메라(camera.py) 지원
   - visualization/renderer.py: 결과(ID, 유사도) 오버레이 표시

  2. 사용 방법

  먼저 필요한 라이브러리를 설치했습니다. (torch, torchvision, timm, scikit-learn, opencv-python 등)

  1) 고양이 등록 (DB 구축)
  디렉토리 구조가 dataset/고양이이름/*.jpg 형태라면 한꺼번에 등록할 수 있습니다.
   1 python main.py --mode register --path ./dataset/cats
  단일 이미지만 등록할 경우:

   1 python main.py --mode register --path ./cat1.jpg --label "Nabi"

  2) 이미지 파일 테스트

   1 python main.py --mode infer --path ./test_image.jpg

  3) 실시간 카메라 테스트
   1 python main.py --mode camera --source 0
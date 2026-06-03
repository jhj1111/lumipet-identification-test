import argparse
import os
import time
import torch
from cat_reid.config import settings
from cat_reid.input.register import CatRegister
from cat_reid.input.file import FileInference
from cat_reid.input.camera import CameraInference
from cat_reid.data.loader import CatDatasetLoader
from cat_reid.evaluation.metrics import Evaluator
from cat_reid.models.extractor import FeatureExtractor
from cat_reid.training.trainer import CatReIdModel, train

def main():
    parser = argparse.ArgumentParser(description="Cat Re-ID Prototype")
    parser.add_argument("--mode", type=str, choices=["register", "infer", "camera", "evaluate", "train"], required=True, help="작동 모드")
    parser.add_argument("--path", type=str, help="이미지 또는 데이터셋 경로")
    parser.add_argument("--label", type=str, help="등록 모드에서 사용할 레이블")
    parser.add_argument("--source", type=int, default=0, help="카메라 소스 인덱스")
    parser.add_argument("--test_size", type=float, default=0.5, help="평가/학습 데이터 비율")
    parser.add_argument("--limit", type=int, help="처리할 고양이 수 제한")
    parser.add_argument("--epochs", type=int, default=5, help="학습 에폭 수")

    args = parser.parse_args()
    custom_extractor = FeatureExtractor(weights_path=settings.LINEAR_MODEL_PATH)

    if args.mode == "register":
        if not args.path:
            print("Error: --path가 필요합니다.")
            return
        register = CatRegister(extractor=custom_extractor)
        if os.path.isdir(args.path):
            register.register_directory(args.path)
        else:
            if not args.label:
                print("Error: 단일 이미지 등록 시 --label이 필요합니다.")
                return
            register.register_image(args.path, args.label)
            register.store.save()
            
    elif args.mode == "infer":
        if not args.path:
            print("Error: --path가 필요합니다.")
            return
        infer = FileInference(extractor=custom_extractor)
        if os.path.isdir(args.path):
            for f in sorted(os.listdir(args.path)):
                f_path = os.path.join(args.path, f)
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label, score = infer.predict(f_path)
                    print(f"File: {f:20s} | Result: {label:10s} | Similarity: {score:.4f}")
        else:
            label, score = infer.predict(args.path)
            print(f"Result: {label} | Similarity: {score:.4f}")

    elif args.mode == "camera":
        camera = CameraInference()
        camera.run(source=args.source)

    elif args.mode == "evaluate":
        if not args.path:
            args.path = settings.DATASET_PATH
        
        loader = CatDatasetLoader(args.path)
        if args.limit:
            loader.cat_folders = loader.cat_folders[:args.limit]
            
        gallery, query = loader.get_split_data(test_size=args.test_size)
        
        print(f"Gallery size: {len(gallery)}, Query size: {len(query)}")
        
        # DB 초기화 및 등록
        if os.path.exists(settings.DB_PATH): os.remove(settings.DB_PATH)
        if os.path.exists(settings.LABEL_PATH): os.remove(settings.LABEL_PATH)
        
        register = CatRegister(extractor=custom_extractor)
        for img_path, label in gallery:
            register.register_image(img_path, label)
        register.store.save()
        
        infer = FileInference(extractor=custom_extractor)
        evaluator = Evaluator(infer)
        evaluator.evaluate(query)

    elif args.mode == "train":
        if not args.path:
            args.path = settings.DATASET_PATH
            
        print(f"Loading dataset for training from {args.path}...")
        loader = CatDatasetLoader(args.path)
        if args.limit:
            loader.cat_folders = loader.cat_folders[:args.limit]
            
        train_loader, val_loader = loader.get_train_loaders(batch_size=16, test_size=0.2)
        
        # 모델 준비
        base_extractor = FeatureExtractor(weights_path=settings.LINEAR_MODEL_PATH)
        model = CatReIdModel(custom_extractor, num_classes=len(loader.cat_folders))
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train(model, train_loader, val_loader, epochs=args.epochs, device=device)
        print("Training completed. Best model saved as best_projection.pth")

if __name__ == "__main__":
    main()

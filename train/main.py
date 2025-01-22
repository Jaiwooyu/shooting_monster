import os
from new_collector import BasketballShotDataCollector
from train_model import main as train_main
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_hours', type=float, default=1.0,
                       help='Number of hours to collect data')
    parser.add_argument('--pretrain', action='store_true',
                       help='Run pre-training with elite players dataset')
    parser.add_argument('--collect_youtube', action='store_true',
                       help='Collect YouTube data')
    parser.add_argument('--finetune', action='store_true',
                       help='Run fine-tuning with YouTube dataset')
    args = parser.parse_args()

    # 사전학습만
    # python main.py --pretrain
    # nohup python main.py --pretrain > pretrain_log.out 2>&1 &


    # YouTube 데이터 수집 및 fine-tuning
    # python main.py --collect_youtube --collection_hours 2 --finetune
    # nohup python main.py --collect_youtube --collection_hours 2 --finetune > finetune_log.out 2>&1 &


    # 전체 과정 (사전학습 -> 데이터 수집 -> fine-tuning)
    # python main.py --pretrain --collect_youtube --collection_hours 2 --finetune
    # nohup python main.py --pretrain --collect_youtube --collection_hours 2 --finetune > full_process_log.out 2>&1 &


    # 1. YouTube 데이터 수집 (선택적)
    if args.collect_youtube:
        search_queries = [
        "kbl 하이라이트",
        # ----------------------------------------
        # 추가 검색어
        # ----------------------------------------
        ]
        
        collector = BasketballShotDataCollector(
            output_dir="basketball_shot_dataset",
            max_training_hours=args.collection_hours,
            search_queries=search_queries,
            max_videos_per_query=50,
            min_resolution=720,
            sequence_length=30
        )
        
        collector.collect_shots(num_threads=4)
    
    # 2. 모델 학습 (사전학습 및 fine-tuning)
    train_main(pretrain=args.pretrain, finetune=args.finetune)

if __name__ == "__main__":
    main()
# test script for inference.py


# Data 무엇을 쓸 것인지 결정: Binary(긍/부정), Multi(긍/부정/중립)
DATA_PATH='../data/binary_data_test.pkl'    # '../data/multi_data_test.pkl'
# 어떤 학습된 모델을 사용할 것인가 (best model & 마지막까지 학습된 모델)
MODEL_PATH='output/best_model/pytorch_model.bin'    # 'output/checkpoint-8500/pytorch_model.bin'
# # 어떤 학습된 arguments를 사용할 것인가 (best model & 마지막까지 학습된 모델)
ARGS_PATH='output/best_model/training_args.bin'    # 'output/checkpoint-8500/training_args.bin'


# 긍/부정 상황을 모두 이용한 Training 및 Inference
'''
Case 1: 긍,부정 & 구질 인과관계 반영
        -> 부정상황일 때 투구구질 Label 변화
Case 2: Transfer Learning
        -> 긍정 상황에서 먼저 학습 → 부정 상황에서 학습
          (부정상황일 때 투구구질 Label 변화 + Transfer learning)
Case 3: 긍, 부정 상황을 Input으로 입력 →  Inference 에서 긍, 부정 상황의 컨트롤을 통한 결과 기대
        -> 부정상황일 때 투구구질 Label 변화하지 않음 + Inference 때 긍, 부정 상황 Control
'''

# Case 1, 2, 3 모델 중에 선택해서 진행


# case 1 모델: 긍,부정 & 구질 인과관계 반영
python scripts/inference.py \
    # Data 무엇을 쓸 것인지 결정: Binary(긍/부정), Multi(긍/부정/중립)
    --data_path=${DATA_PATH}
    --model_path=${MODEL_PATH} \
    --args_path=${ARGS_PATH} 


# case 2 모델: Transfer Learning
python scripts/inference.py \
    # Data 무엇을 쓸 것인지 결정: Binary(긍/부정), Multi(긍/부정/중립)
    --data_path=${DATA_PATH}
    --model_path=${MODEL_PATH} \
    --args_path=${ARGS_PATH}

# case 3 모델: 긍, 부정 상황을 Input으로 입력 →  Inference 에서 긍, 부정 상황의 컨트롤을 통한 결과 기대
# 기존 긍,부정 상황일 때와 반대로 했을 때 "2 번" Inference 진행해야 함
python scripts/inference.py \
    # Data 무엇을 쓸 것인지 결정: Binary(긍/부정), Multi(긍/부정/중립)
    --data_path=${DATA_PATH}
    --model_path=${MODEL_PATH} \
    --args_path=${ARGS_PATH}
    # TODO: --label_switch_inference 없는 것도 Inference 진행할 것 (총 2번)
    --label_switch_inference  # 해당 부분은 긍,부정을 변경했을 때 여부
# test script for train.py


# 긍/부정 상황을 모두 이용한 학습
'''
Case 1: 긍,부정 & 구질 인과관계 반영
        -> 부정상황일 때 투구구질 Label 변화
Case 2: Transfer Learning
        -> 긍정 상황에서 먼저 학습 → 부정 상황에서 학습
          (부정상황일 때 투구구질 Label 변화 + Transfer learning)
Case 3: 긍, 부정 상황을 Input으로 입력 →  Inference 에서 긍, 부정 상황의 컨트롤을 통한 결과 기대
        -> 부정상황일 때 투구구질 Label 변화하지 않음 + Inference 때 긍, 부정 상황 Control
'''

# Data 무엇을 쓸 것인지 결정: Binary(긍/부정), Multi(긍/부정/중립)
DATA_PATH='../data/binary_data_train.pkl'    # '../data/multi_data_train.pkl'
EVAL_DATA_PATH='../data/binary_data_val.pkl' # '../data/multi_data_val.pkl'
OUTPUT_PATH='./output/'
TRAIN_BATCH_SIZE=512
EVAL_BATCH_SIZE=1024
EPOCHS=30
LR=1e-3

N_ENCODER=4
N_DECODER=4
N_CONCAT=4
D_MODEL=64
N_HEAD=4
D_FEEDFORWARD=256

# Sampling 빈도 설정
'''
picther -> <투구구질>만 동등하게 (7 가지)
batter -> <긍,부정 상황> 동등하게 (2 가지)
both -> <투구구질 & 긍,부정 상황> 모두 동등하게 (14 가지) 
'''
SAMPLE_CRITERIA='pitcher'



# Case 1, 2, 3 모델 중에 선택해서 진행


# case 1 모델: 긍,부정 & 구질 인과관계 반영
python scripts/train.py \
    # train data 경로
    --train_data_file=${DATA_PATH} \
    # validation data 경로
    --eval_data_file=${EVAL_DATA_PATH} \
    # output 저장 경로
    --output_dir=${OUTPUT_PATH} \
    # train 할 때, batch size
    --train_batch_size=${TRAIN_BATCH_SIZE} \
    # validation 할 때, batch size
    --eval_batch_size=${EVAL_BATCH_SIZE} \
    # train 할 때, epoch 수
    --num_train_epochs=${EPOCHS} \
    # train 할 때, learning rate
    --learning_rate=${LR} \
    # 모델의 Encoder layer 중첩 수
    --n_encoder_layer=${N_ENCODER} \
    # 모델의 Decoder layer 중첩 수
    --n_decoder_layer=${N_DECODER} \
    # 현재 모델에는 필요없는 변수 (Multi-task 모델에서 사용됨: <타격 예측, 구질 예측>)
    --n_concat_layer=${N_CONCAT} \
    # input의 차원 수
    --d_model=${D_MODEL} \
    # multi-head attention에서 head의 수
    --nhead=${N_HEAD} \
    # feedforward network의 차원 수
    --dim_feedforward=${D_FEEDFORWARD} \
    # input의 Continuous 변수 중 Stats 변수들(단기, 중기, 장기)을 각각 group 지어 embedding 할 것인지 여부
    --grouping \
    # train 중에 Log 나타낼 step의 수 (학습 중에 evaluation을 한다면!)
    --logging_step=500 \
    # train 중에 evaluation을 진행할 것인지
    --evaluate_during_training \
    # output directory에 덮어써서 저장할 것인지
    --overwrite_output_dir \
    # 학습 중인지 여부
    --do_train \
    # sampling 빈도 설정: <1. 투구구질 기준, 2. 긍,부정 상황 기준, 3. 모두 고려>
    --sample_criteria=${SAMPLE_CRITERIA} 


'''
# case 2 모델: Transfer Learning
python scripts/train.py \
    # train data 경로
    --train_data_file=${DATA_PATH} \
    # validation data 경로
    --eval_data_file=${EVAL_DATA_PATH} \
    # output 저장 경로
    --output_dir=${OUTPUT_PATH} \
    # train 할 때, batch size
    --train_batch_size=${TRAIN_BATCH_SIZE} \
    # validation 할 때, batch size
    --eval_batch_size=${EVAL_BATCH_SIZE} \
    # train 할 때, epoch 수
    --num_train_epochs=${EPOCHS} \
    # train 할 때, learning rate
    --learning_rate=${LR} \
    # 모델의 Encoder layer 중첩 수
    --n_encoder_layer=${N_ENCODER} \
    # 모델의 Decoder layer 중첩 수
    --n_decoder_layer=${N_DECODER} \
    # 현재 모델에는 필요없는 변수 (Multi-task 모델에서 사용됨: <타격 예측, 구질 예측>)
    --n_concat_layer=${N_CONCAT} \
    # input의 차원 수
    --d_model=${D_MODEL} \
    # multi-head attention에서 head의 수
    --nhead=${N_HEAD} \
    # feedforward network의 차원 수
    --dim_feedforward=${D_FEEDFORWARD} \
    # input의 Continuous 변수 중 Stats 변수들(단기, 중기, 장기)을 각각 group 지어 embedding 할 것인지 여부
    --grouping \
    # train 중에 Log 나타낼 step의 수 (학습 중에 evaluation을 한다면!)
    --logging_step=500 \
    # train 중에 evaluation을 진행할 것인지
    --evaluate_during_training \
    # output directory에 덮어써서 저장할 것인지
    --overwrite_output_dir \
    # 학습 중인지 여부
    --do_train \
    # sampling 빈도 설정: <1. 투구구질 기준, 2. 긍,부정 상황 기준, 3. 모두 고려>
    --sample_criteria=${SAMPLE_CRITERIA}
    # case 2 모델로 Transfer learning 진행
    --do_transfer


# case 3 모델: 긍, 부정 상황을 Input으로 입력 →  Inference 에서 긍, 부정 상황의 컨트롤을 통한 결과 기대
python scripts/train.py \
    # train data 경로
    --train_data_file=${DATA_PATH} \
    # validation data 경로
    --eval_data_file=${EVAL_DATA_PATH} \
    # output 저장 경로
    --output_dir=${OUTPUT_PATH} \
    # train 할 때, batch size
    --train_batch_size=${TRAIN_BATCH_SIZE} \
    # validation 할 때, batch size
    --eval_batch_size=${EVAL_BATCH_SIZE} \
    # train 할 때, epoch 수
    --num_train_epochs=${EPOCHS} \
    # train 할 때, learning rate
    --learning_rate=${LR} \
    # 모델의 Encoder layer 중첩 수
    --n_encoder_layer=${N_ENCODER} \
    # 모델의 Decoder layer 중첩 수
    --n_decoder_layer=${N_DECODER} \
    # 현재 모델에는 필요없는 변수 (Multi-task 모델에서 사용됨: <타격 예측, 구질 예측>)
    --n_concat_layer=${N_CONCAT} \
    # input의 차원 수
    --d_model=${D_MODEL} \
    # multi-head attention에서 head의 수
    --nhead=${N_HEAD} \
    # feedforward network의 차원 수
    --dim_feedforward=${D_FEEDFORWARD} \
    # input의 Continuous 변수 중 Stats 변수들(단기, 중기, 장기)을 각각 group 지어 embedding 할 것인지 여부
    --grouping \
    # train 중에 Log 나타낼 step의 수 (학습 중에 evaluation을 한다면!)
    --logging_step=500 \
    # train 중에 evaluation을 진행할 것인지
    --evaluate_during_training \
    # output directory에 덮어써서 저장할 것인지
    --overwrite_output_dir \
    # 학습 중인지 여부
    --do_train \
    # sampling 빈도 설정: <1. 투구구질 기준, 2. 긍,부정 상황 기준, 3. 모두 고려>
    --sample_criteria=${SAMPLE_CRITERIA}
    # input에 pos/neg 정보 넣을지에 대한 여부 (모델 변경)
    --is_sentiment_input
'''
eval_results_dir='./exp/results.csv'

DATA_PATH='./data/binary_memory_50/binary_data_memory_train(all).pkl'
EVAL_DATA_PATH='./data/binary_memory_50/binary_data_memory_val.pkl'
TRAIN_BATCH_SIZE=512
EVAL_BATCH_SIZE=1024
EPOCHS=200

N_PITCHER_DISCRETE=12
N_PITCHER_CONTINUOUS=19
N_BATTER_DISCRETE=2
N_BATTER_CONTINUOUS=18

N_MEMORY=12
N_ENCODER=12
D_MODEL=256
N_HEAD=1

LOGGING_STEP=100

DECAY=1e-5
MEMORY_LEN=50
LR=5e-4
WEIGHTED_SAMPLING=0

MAX_STEPS=3000
for ATTENTION in 'dot'; do
    for TRAIN_BATCH_SIZE in 768; do
        for N_HEAD in 1; do
            for MEMORY_LEN in 50; do
                for D_MODEL in 128; do
                    for N_ENCODER in 6; do
                        for N_MEMORY in 6; do

                            NAME='batch'${TRAIN_BATCH_SIZE}'_lr'${LR}'_encodernum'${N_ENCODER}'_memorynum'${N_MEMORY}'_modeldim'${D_MODEL}'_memorylen'${MEMORY_LEN}'_head'${N_HEAD}'_attn_'${ATTENTION}
                            TB_PATH='./runs/'${NAME}
                            OUTPUT_PATH='./output/memory_model/'${NAME}
                            mkdir -p ${OUTPUT_PATH}
                            python src_baseballtransformer/train.py \
                                --train_data_file=${DATA_PATH} \
                                --eval_data_file=${EVAL_DATA_PATH} \
                                --output_dir=${OUTPUT_PATH} \
                                --n_pitcher_disc=${N_PITCHER_DISCRETE} \
                                --n_pitcher_cont=${N_PITCHER_CONTINUOUS} \
                                --n_batter_disc=${N_BATTER_DISCRETE} \
                                --n_batter_cont=${N_BATTER_CONTINUOUS} \
                                --n_memory_layer=${N_MEMORY} \
                                --n_encoder_layer=${N_ENCODER} \
                                --d_model=${D_MODEL} \
                                --n_head=${N_HEAD} \
                                --memory_len=${MEMORY_LEN} \
                                --train_batch_size=${TRAIN_BATCH_SIZE} \
                                --eval_batch_size=${EVAL_BATCH_SIZE} \
                                --max_steps=${MAX_STEPS} \
                                --learning_rate=${LR} \
                                --weight_decay=${DECAY} \
                                --logging_step=${LOGGING_STEP} \
                                --tb_writer_dir=${TB_PATH} \
                                --evaluate_during_training \
                                --overwrite_output_dir \
                                --do_train \
                                --attention_type=${ATTENTION} \
                                --warmup_percent=0.1 \
                                --weighted_sampling=${WEIGHTED_SAMPLING} \
                                --fp16

                            MODEL_PATH=${OUTPUT_PATH}'/best_model/pytorch_model.bin'
                            ARGS_PATH=${OUTPUT_PATH}'/best_model/training_args.bin'

                            python src_baseballtransformer/inference.py \
                                --model_path=${MODEL_PATH} \
                                --args_path=${ARGS_PATH} 
                        done
                    done
                done
            done
        done
    done
done
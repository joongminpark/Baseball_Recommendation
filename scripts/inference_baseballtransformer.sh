MODEL_PATH='./output/memory_model/best_model/pytorch_model.bin'
ARGS_PATH='./output/memory_model/best_model/training_args.bin'

python src_baseballtransformer/inference.py \
    --model_path=${MODEL_PATH} \
    --args_path=${ARGS_PATH} 
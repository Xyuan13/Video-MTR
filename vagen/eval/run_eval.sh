#!/bin/bash
set -e
echo "🚀 Running evaluation"
# Configuration
export VIDEOENV_IMAGE_FACTOR=28 # We increase the factor to 40 for the evaluation of VideoMME dataset for finer-grained inference.
echo "  * VIDEOENV_IMAGE_FACTOR: $VIDEOENV_IMAGE_FACTOR"

MODEL_PATH="/mnt/jfs/Video-MTR" # Add the path to the model you want to evaluate, which should be in the huggingface format.
DATA_ROOT="/mnt/jfs/Video-Datasets" # Add the path to the data root of the video datasets, which contain the video datasets in the following subdirectories: VideoMME/MLVU_Test
EXPERIMENT_BASE_NAME="eval_test"

DATASETS=("mlvu_test") # # dataset name must match annotation file (eval_mlvu_test.json)


echo "📂 Model path: $MODEL_PATH"
echo "🏷️ Eval experiment name: $EXPERIMENT_BASE_NAME"
echo "📊 Datasets: ${DATASETS[@]}"

# Run the evaluation using real training components
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m vagen.eval.eval_bench_video_env \
    --model_path "$MODEL_PATH" \
    --file_name "if$VIDEOENV_IMAGE_FACTOR" \
    --datasets "${DATASETS[@]}" \
    --data_root "$DATA_ROOT" \
    --prefix $EXPERIMENT_BASE_NAME \
    --eval_fresh \


echo "✅ Evaluation completed! Check the results in ./results/ directory"
echo "📄 Full log saved to eval_training_components.log" 

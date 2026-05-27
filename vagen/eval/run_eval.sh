#!/bin/bash
set -e

# Resolve real libcuda/libnvidia-ml — required when the container ships only the
# CUDA stub library (Triton kernels emit "undefined symbol: cuModuleGetFunction"
# without this). Safe no-op on hosts that already expose the real driver.
_EVAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_EVAL_SCRIPT_DIR}/nvidia_driver_preload.inc.sh"

echo "🚀 Running evaluation"
# Configuration
# VIDEOENV_IMAGE_FACTOR: spatial token granularity (higher → more vision tokens per frame).
export VIDEOENV_IMAGE_FACTOR=28
echo "  * VIDEOENV_IMAGE_FACTOR: $VIDEOENV_IMAGE_FACTOR"
# VIDEOEVAL_MAX_MODEL_LEN: vLLM context length. 32768 fits 80-frame inference; reduce to 24576 if OOM.
export VIDEOEVAL_MAX_MODEL_LEN="${VIDEOEVAL_MAX_MODEL_LEN:-32768}"
echo "  * VIDEOEVAL_MAX_MODEL_LEN: $VIDEOEVAL_MAX_MODEL_LEN"

# Override MODEL_PATH / DATA_ROOT via environment to point at your own paths.
MODEL_PATH="${MODEL_PATH:-/mnt/jfs/Video-MTR}" # Modify the path to the model you want to evaluate, which should be in the huggingface format.
DATA_ROOT="${DATA_ROOT:-/mnt/jfs/Video-Datasets}" # Modify the path to the data root of the video datasets, which contain the video datasets in the following subdirectories: VideoMME/MLVU_Test
EXPERIMENT_BASE_NAME="${EXPERIMENT_BASE_NAME:-eval_test}"

DATASETS=(${DATASETS:-mlvu_test}) # dataset name(s) must match annotation file (eval_<name>.json); override via env: DATASETS="videomme_long videomme_medium"


echo "📂 Model path: $MODEL_PATH"
echo "🏷️ Eval experiment name: $EXPERIMENT_BASE_NAME"
echo "📊 Datasets: ${DATASETS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m vagen.eval.eval_bench_video_env \
    --model_path "$MODEL_PATH" \
    --file_name "if$VIDEOENV_IMAGE_FACTOR" \
    --datasets "${DATASETS[@]}" \
    --data_root "$DATA_ROOT" \
    --prefix $EXPERIMENT_BASE_NAME \
    --eval_fresh \


echo "✅ Evaluation completed! Check the results in ./results/ directory"
echo "📄 Full log saved to eval_training_components.log"

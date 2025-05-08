#!/bin/bash

# ==============================================================================
# DeepFakeDetection experiment script
# ==============================================================================
#
# Usage: 
#   ./run_experiment.sh [extra_args_for_train.py]
#
# Description:
#   This script is used for preprocessing video datasets (DFDC or FF++) and 
#   for training a model using the specified configuration.
#
# Note:
#   Ensure that the script has execute permissions: chmod +x run_experiment.sh
#
# ==============================================================================

DATASET_TYPE="ffpp"
CONFIG_FILE="config/xception3d_ffpp.yaml"

PREPROCESS_OUTPUT_DIR="data_frames"

#DFDC_METADATA_PATH="data/metadata.json"
#DFDC_VIDEO_DIR="data/dfdc_all_videos"

FFPP_ORIG_ROOT="data/ff++/original_sequences/actors/c23/videos/original"
FFPP_MANIP_ROOT="data/ff++/manipulated_sequences/DeepFakeDetection/c23/videos"
FFPP_VAL_SIZE=0.15
FFPP_TEST_SIZE=0.15

NUM_FRAMES_EXTRACT=30
FORCE_PREPROCESS=false
SKIP_PREPROCESS=true

# End Configuration

set -e

echo "================================================="
echo " Starting Experiment"
echo "================================================="
echo " Dataset Type:    ${DATASET_TYPE}"
echo " Config File:     ${CONFIG_FILE}"
echo " Frame Output Dir: ${PREPROCESS_OUTPUT_DIR}"
echo "-------------------------------------------------"

# STEP 1: Preprocessing (Optional)
if [ "${SKIP_PREPROCESS}" = true ]; then
    echo "[INFO] STEP 1/2: Preprocessing SKIPPED as requested."
else
    echo "[INFO] STEP 1/2: Starting Preprocessing (Dataset: ${DATASET_TYPE})..."

    PREPROCESS_ARGS=(
        --dataset "${DATASET_TYPE}"
        --output_dir "${PREPROCESS_OUTPUT_DIR}"
        --num_frames "${NUM_FRAMES_EXTRACT}"
    )
    if [ "${FORCE_PREPROCESS}" = false ]; then
        PREPROCESS_ARGS+=("--skip_existing")
        echo "[INFO] --skip_existing enabled."
    else
        echo "[WARN] FORCE_PREPROCESS enabled, existing frames may be overwritten/ignored."
    fi

    if [ "${DATASET_TYPE}" = "dfdc" ]; then
        PREPROCESS_ARGS+=(
            "--dfdc_metadata_path" "${DFDC_METADATA_PATH}"
            "--dfdc_video_dir" "${DFDC_VIDEO_DIR}"
        )
    elif [ "${DATASET_TYPE}" = "ffpp" ]; then
        PREPROCESS_ARGS+=(
            "--ffpp_orig_root" "${FFPP_ORIG_ROOT}"
            "--ffpp_manip_root" "${FFPP_MANIP_ROOT}"
            "--ffpp_val_size" "${FFPP_VAL_SIZE}"
            "--ffpp_test_size" "${FFPP_TEST_SIZE}"
        )
    else
        echo "[ERROR] Unrecognized dataset type: ${DATASET_TYPE}"
        exit 1
    fi

    python scripts/preprocess_frames.py "${PREPROCESS_ARGS[@]}"

    echo "[INFO] STEP 1/2: Preprocessing completed."
fi
echo "-------------------------------------------------"

# STEP 2: Training
echo "[INFO] STEP 2/2: Starting Training..."

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "[ERROR] Training configuration file not found: ${CONFIG_FILE}"
    exit 1
fi

echo "[INFO] Using configuration: ${CONFIG_FILE}"

TRAIN_ARGS=(
    "--config" "${CONFIG_FILE}"
    "$@" # Pass extra args
)

python src/train.py "${TRAIN_ARGS[@]}"

echo "[INFO] STEP 2/2: Training completed."
echo "================================================="
echo " Experiment Finished"
echo "================================================="

exit 0

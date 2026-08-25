#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${PROTCHORD_DATASET:?set PROTCHORD_DATASET to an LMDB test set}"
MODEL_CKPT="${PROTCHORD_CHECKPOINT:-$ROOT_DIR/checkpoints/ProtChord_RL.pt}"
LLM_DIR="${PROTCHORD_LLM_DIR:-$ROOT_DIR/config}"
TOKENIZER_DIR="${PROTCHORD_TOKENIZER_DIR:-$ROOT_DIR/tokenizer}"
RESULT_PATH="${PROTCHORD_RESULT_PATH:-$ROOT_DIR/results/inference.pkl}"

mkdir -p "$(dirname "$RESULT_PATH")"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

exec python -m protchord.infer_complex \
  --strategy Single \
  --stage "${PROTCHORD_STAGE:-stageB}" \
  --valid_data_path "$DATASET" \
  --train_data_path "$DATASET" \
  --pretrained_ckpt_path "$MODEL_CKPT" \
  --adapter_ckpt_path "$MODEL_CKPT" \
  --vae_ckpt_path "$MODEL_CKPT" \
  --llm_model_name_or_path "$LLM_DIR" \
  --tokenizer_path "$TOKENIZER_DIR" \
  --infer_result_path "$RESULT_PATH" \
  --load_ckpt \
  --temperature "${PROTCHORD_TEMPERATURE:-1.5}" \
  --model_max_length "${PROTCHORD_MODEL_MAX_LENGTH:-2048}" \
  --protein_max_size "${PROTCHORD_PROTEIN_MAX_SIZE:-2048}" \
  --backbone "${PROTCHORD_BACKBONE:-exp3}" \
  --encoder_attention_heads 32 \
  --encoder_layers 32 \
  --encoder_ffn_embed_dim 8192 \
  --encoder_embed_dim 2048 \
  --decoder_hidden_dim 2048 \
  --decoder_ffn_dim 8192 \
  --encoder_pair_embed_dim 64 \
  --num_pred_attn_layer 16 \
  --num_3d_bias_kernel 32 \
  --num_edges 25600 \
  --equivar_vec_init RELATIVE_POS_VEC_BIAS \
  --fp16 \
  --use_memory_efficient_attention \
  --mask_ratio 0 \
  --only_use_rotary_embedding_for_protein

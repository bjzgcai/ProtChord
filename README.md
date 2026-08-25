# ProtChord

ProtChord is a protein-pocket-to-molecule generation model. The release contains the inference code, the public tokenizer, and the CrossDocked2020 test split used by the inference workflow. The inference checkpoint and training data are distributed separately.

## Installation

With a CUDA-enabled Python environment active:

```bash
python -m pip install -r requirements.txt
```

The requirements intentionally contain no local editable training-framework package.

## Model and tokenizer

The decoder starts from the LLaMA-3 vocabulary and uses a compact domain vocabulary for molecular SMILES and protein sequences: 1,401 molecular tokens, the 26 protein alphabet tokens, and modality/control markers for molecule and protein spans. The released tokenizer and checkpoint are already aligned.

The model configuration is in `config/`. The inference checkpoint is named `ProtChord_RL.pt` and should be placed at `checkpoints/ProtChord_RL.pt`. The `checkpoints/` directory is ignored by default because large model files are distributed separately.

## Data

The inference test pickle is in `data/`:

- `crossdocked2020_test.pkl`

The SFT and DPO training pickles are intentionally excluded from this repository and are not required to run inference. Obtain them separately if you need to reproduce training.

## Build the test LMDB

With the required Python environment active:

```bash
python process_data.py \
  --test-pkl data/crossdocked2020_test.pkl \
  --tokenizer tokenizer \
  --output-lmdb data/crossdocked2020_test.lmdb
```

This writes the LMDB format consumed by the inference dataset reader.

## Run inference

Place `ProtChord_RL.pt` under `checkpoints/`, then run:

```bash
export PROTCHORD_DATASET=$PWD/data/crossdocked2020_test.lmdb
export PROTCHORD_CHECKPOINT=$PWD/checkpoints/ProtChord_RL.pt
bash scripts/infer_complex.sh
```

The script uses the bundled `config/` and `tokenizer/` by default. Set `PROTCHORD_RESULT_PATH` to choose the output pickle, and use `PROTCHORD_MAX_SAMPLES`, `PROTCHORD_NUM_SAMPLES`, and `PROTCHORD_NUM_RETURN_SEQUENCES` to control the amount of inference.

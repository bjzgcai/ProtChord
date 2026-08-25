# ProtChord

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22093629.svg)](https://doi.org/10.5281/zenodo.22093629)

ProtChord is a protein-pocket-to-molecule generation model. This repository contains the inference code, model configuration, public tokenizer, and the CrossDocked2020 test split used by the inference workflow.

The ProtChord-RL checkpoint, processed training data, and raw evaluation results are archived separately on Zenodo.

## Installation

With a CUDA-enabled Python environment active:

```bash
python -m pip install -r requirements.txt
```

The requirements intentionally contain no local editable training-framework package.

## Model and tokenizer

The decoder starts from the LLaMA-3 vocabulary and uses a compact domain vocabulary for molecular SMILES and protein sequences: 1,401 molecular tokens, the 26 protein alphabet tokens, and modality/control markers for molecule and protein spans. The released tokenizer and checkpoint are already aligned.

The model configuration is provided in `config/`.

The inference checkpoint is named:

```text
ProtChord_RL.pt
```

and should be placed at:

```text
checkpoints/ProtChord_RL.pt
```

The `checkpoints/` directory is ignored by default because large model files are distributed separately.

## Checkpoint and research artifacts

The released ProtChord-RL checkpoint, processed CrossDocked2020 data, and raw molecular-generation results are archived on Zenodo:

**DOI:** [10.5281/zenodo.22093629](https://doi.org/10.5281/zenodo.22093629)

### Checkpoint

For reliable transfer, the ProtChord-RL checkpoint is distributed as four sequential parts:

```text
ProtChord_RL.pt.part-00
ProtChord_RL.pt.part-01
ProtChord_RL.pt.part-02
ProtChord_RL.pt.part-03
```

Download all four parts from Zenodo into the same directory and reconstruct the original checkpoint with:

```bash
cat ProtChord_RL.pt.part-* > ProtChord_RL.pt
```

The reconstructed checkpoint must have the following SHA256 checksum:

```text
95bbe5e288c72a6eaade1f59b67263eed61cbd259ef6652108b951231bf7ceb7
```

Verify the reconstructed checkpoint with:

```bash
sha256sum ProtChord_RL.pt
```

After verification, place the checkpoint at:

```text
checkpoints/ProtChord_RL.pt
```

### Processed data

The Zenodo release contains the processed CrossDocked2020 data used in the study:

```text
crossdocked2020_sft.pkl
crossdocked2020_dpo.pkl
crossdocked2020_test.pkl
```

- `crossdocked2020_sft.pkl`: processed data used for Stage B supervised fine-tuning.
- `crossdocked2020_dpo.pkl`: processed data used for Stage C preference optimization with DPO.
- `crossdocked2020_test.pkl`: processed CrossDocked2020 evaluation data.

### Evaluation results

The Zenodo release also contains the archived molecular-generation results:

```text
ProtChord_Results.pkl
ProtChord_RL_Results.pkl
```

- `ProtChord_Results.pkl`: raw generation/evaluation outputs associated with the base ProtChord model.
- `ProtChord_RL_Results.pkl`: raw generation/evaluation outputs associated with ProtChord-RL.

The Zenodo release additionally provides `SHA256SUMS.txt` for file-integrity verification.

## Data

The CrossDocked2020 test pickle required by the inference workflow is included directly in this repository:

```text
data/crossdocked2020_test.pkl
```

The SFT and DPO training pickles are intentionally excluded from the GitHub repository because they are not required for inference. They are available from the associated Zenodo release:

https://doi.org/10.5281/zenodo.22093629

## Build the test LMDB

The inference dataset reader consumes an LMDB file. Build it from the released test pickle with:

```bash
python process_data.py \
  --test-pkl data/crossdocked2020_test.pkl \
  --tokenizer tokenizer \
  --output-lmdb data/crossdocked2020_test.lmdb
```

This creates:

```text
data/crossdocked2020_test.lmdb
```

## Run inference

First reconstruct and place the ProtChord-RL checkpoint at:

```text
checkpoints/ProtChord_RL.pt
```

Then set the dataset and checkpoint paths:

```bash
export PROTCHORD_DATASET=$PWD/data/crossdocked2020_test.lmdb
export PROTCHORD_CHECKPOINT=$PWD/checkpoints/ProtChord_RL.pt
```

Run inference with:

```bash
bash scripts/infer_complex.sh
```

The inference script uses the bundled `config/` and `tokenizer/` by default.

The following environment variables can be used to control inference:

- `PROTCHORD_RESULT_PATH`: output pickle path.
- `PROTCHORD_MAX_SAMPLES`: maximum number of test samples to process.
- `PROTCHORD_NUM_SAMPLES`: number of samples generated during inference.
- `PROTCHORD_NUM_RETURN_SEQUENCES`: number of returned sequences.

For example:

```bash
export PROTCHORD_RESULT_PATH=$PWD/results/ProtChord_RL_Results.pkl
export PROTCHORD_MAX_SAMPLES=100
export PROTCHORD_NUM_SAMPLES=10
export PROTCHORD_NUM_RETURN_SEQUENCES=10

bash scripts/infer_complex.sh
```

<!-- ## File integrity

The Zenodo artifact includes SHA256 checksums for the released binary files.

After downloading the files, they can be verified with:

```bash
sha256sum -c SHA256SUMS.txt
```

For the reconstructed checkpoint, the expected SHA256 is:

```text
95bbe5e288c72a6eaade1f59b67263eed61cbd259ef6652108b951231bf7ceb7
```

## Research artifacts

The model checkpoint, processed training/evaluation data, and raw evaluation outputs associated with ProtChord are available at:

**Zenodo:** https://doi.org/10.5281/zenodo.22093629

**Source code:** https://github.com/weizhang-ustc/ProtChord

## Citation

If you use ProtChord, please cite the accompanying paper:

**ProtChord: Structure-Sequence Alignment for Protein-Centered Discovery**

Please also cite the released research artifacts when using the checkpoint, processed data, or evaluation outputs:

**ProtChord Research Artifacts**  
Zenodo DOI: [10.5281/zenodo.22093629](https://doi.org/10.5281/zenodo.22093629) -->
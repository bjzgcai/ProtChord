export PROTCHORD_DATASET=$PWD/data/crossdocked2020_test.lmdb
export PROTCHORD_CHECKPOINT=$PWD/checkpoints/ProtChord_RL.pt
export PROTCHORD_MAX_SAMPLES=100
export PROTCHORD_NUM_SAMPLES=100
export PROTCHORD_NUM_RETURN_SEQUENCES=100
export PROTCHORD_RESULT_PATH=$PWD/results/crossdocked_testset_100x100.pkl

bash scripts/infer_complex.sh
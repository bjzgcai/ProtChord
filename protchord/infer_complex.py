# -*- coding: utf-8 -*-
import os
import sys
from typing import Dict
from dataclasses import dataclass


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

import pickle

import hydra
from omegaconf import MISSING, DictConfig, OmegaConf
from tqdm import tqdm

# from protchord_core.data.prot_data.prottext_dataset import ProteinTextDataset
from protchord_core.data.prot_data.complex_stageB_dataset import ComplexStageBDataset
from protchord.tokenizer import ProtChordTokenizer
from protchord_core.logging import logger
from protchord_core.models.progpt.complexgpt import ComplexGPTModel
from protchord_core.models.encoder.encoder_config import EncoderConfig
from protchord_core.pipeline.accelerator.dataclasses import DistributedTrainConfig, TrainStrategy
from protchord_core.utils.cli_utils import cli

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
import torch

torch.set_printoptions(edgeitems=torch.inf)


def make_supervised_data_module(args, mode="train") -> Dict:
    assert mode in [
        "train",
        "eval",
        "test",
    ], f"Invalid mode: {mode}, must be train, eval, or test."

    tokenizer = ProtChordTokenizer.from_pretrained(args.tokenizer_path)
    args.vocab_size = len(tokenizer)
    args.pad_token_id = tokenizer.pad_token_id

    """ Make dataset and collator for supervised fine-tuning. """

    if args.stage != "stageB":
        raise ValueError("ProtChord release supports stageB complex inference only")
    datasetclass = ComplexStageBDataset

    valid_dataset = datasetclass(
        data_path=args.valid_data_path,
        model_max_length=4096,
        protein_max_size=args.protein_max_size,
        pad_token_id=tokenizer.pad_token_id,
        pool_mode=args.pool_mode,
        embedding_length=args.embedding_length,
        num_token_id=tokenizer.encode("<num>", add_special_tokens=False)[0],
        protein_pad_id=1,
        pp_mode=(
            args.strategy == TrainStrategy.Pipeline
            or args.strategy == TrainStrategy.ThreeD
        ),
        local_rank=args.local_rank,
        use_llama_tokenizer=args.use_llama_tokenizer,
        instruction_mode=args.instruction_mode,
        use_vae=args.use_vae,
    )
    for idx in range(len(valid_dataset)):
        valid_dataset[idx]
    return dict(
        train_dataset=None,
        eval_dataset=valid_dataset,
        vocab_size=len(tokenizer),
        tokenizer=tokenizer,
    )




# Inference-only CLI configuration.
@dataclass
class ComplexGPTConfig:
    model_type = "complexgpt"

    dataset_splits: str = ""
    dataset_ratios: str = ""
    is_progpt: bool = True

    pool_mode: str = "full"
    embedding_length: int = 20
    model_max_length: int = 512

    smiles_dict_path: str = ""
    encoder_checkpoint_path: str = ""
    llm_model_name_or_path: str = ""
    mol_size_path: str = ""
    tokenizer_path: str = ""
    pretrained_ckpt_path: str = ""

    mfm_lora: bool = False
    llm_lora: bool = False
    btn_adaptor: bool = False

    fused_graphormer_llama: bool = False
    add_mol_attn_bias_in_llama: bool = False
    mol_attn_bias_in_llama_layerwise: bool = False
    path_edge_cutoff: int = 0

    skip_num_datasets: str = ""
    num_data_loading_workers: int = 16
    use_global_padding: bool = False
    max_num_mol_per_sample: int = 8
    protein_max_size: int = 2048 # 2048

    # add use llama tokenizer
    use_llama_tokenizer: bool = False
    instruction_mode: bool = False

    train_data_path: str = ""
    valid_data_path: str = ""
    epochs: int = 15
    backbone: str = "dit"


    # adapter
    adapter_ckpt_path: str = ""

    # save
    infer_result_path: str = ""
    # infer
    temperature: float = 0.75

    stage: str = "stageA"
    # VAE
    use_vae: bool = False
    vae_latent_size: int = 2048
    vae_kl_weight: float = 0.001
    vae_beta: float = 0.1
    vae_ckpt_path: str = ""
    
    add_noise: bool = False # for inference
    infer_noise_beta: float = 1.0
    train_noise_beta: float = 1.0

    # DPO
    use_dpo: bool = False
    dpo_beta: float = 0.1
    dpo_freeze_update_steps: int = 100  # 每 K 步把 policy 同步到 ref
    dpo_sample_num: int = 20
    dpo_n_candidates: int = 5   # 每个 prompt 采样的候选数
    dpo_max_new_tokens: int = 256
    dpo_temperature: float = 1.0
    dpo_top_p: float = 0.9
    dpo_offline_data_path: str = ""
    dpo_seed: int = 66
    dpo_sort_key: str = "vina_dock"
    dpo_mode: str = "online"
    dpo_fused_ring_reward: float = 0.5
    # gradient_accumulation_steps: int = 1

@cli(
    DistributedTrainConfig,
    EncoderConfig,
    ComplexGPTConfig,
)
@hydra.main(
    version_base=None, config_path="../../config", config_name="config_encoder"
)
def main(args) -> None:
    data_module = make_supervised_data_module(args, mode="test")
    logger.info(f"vocab size: {data_module['vocab_size']}")
    if not args.encoder_checkpoint_path:
        args.encoder_checkpoint_path = args.pretrained_ckpt_path
    model = ComplexGPTModel(args, data_module["vocab_size"])
    tokenizer = data_module["tokenizer"]
    print(f"temperature:{args.temperature}")
    model.cuda()
    model.load_ckpt()
    model.eval()
    valid_dataset = data_module["eval_dataset"]
    collate_fn = data_module["eval_dataset"].collate_fn  # 获取 collate 函数
    batch_size = 1
    sample_each_target = int(os.environ.get("PROTCHORD_NUM_SAMPLES", "120"))
    sample_each_time = int(os.environ.get("PROTCHORD_NUM_RETURN_SEQUENCES", "120"))
    result = []
    max_samples = int(os.environ.get("PROTCHORD_MAX_SAMPLES", "0"))
    total_samples = min(len(valid_dataset), max_samples) if max_samples > 0 else len(valid_dataset)
    for i in tqdm(range(0, total_samples, batch_size), desc="Processing Batches"):
        batch_samples = [
            valid_dataset[j] for j in range(i, min(i + batch_size, len(valid_dataset)))
        ]
        batch = collate_fn(batch_samples)
        cur_result = []
        batch[0][-1]["token_id"][0].numpy()
        # cur_result.append(get_aa_by_id(token_id))
        tmp_id = []
        for i in batch[0][0][0]:
            if i > 0:
                tmp_id.append(i)

        decode_output = tokenizer.decode(tmp_id)
        cur_result.append(decode_output)
        cur_result.append(batch_samples[0][1]['pl_tuple'])

        tmp = []
        cur_smi_set = set()
        for j in range(sample_each_target // sample_each_time):
            out = model.generate(
                batch,
                max_new_tokens=256, # 256,
                num_return_sequences=sample_each_time,
                output_scores=False,
                do_sample=os.environ.get("PROTCHORD_DO_SAMPLE", "1") != "0",
                top_p=float(os.environ.get("PROTCHORD_TOP_P", "0.95")),
                temperature=args.temperature,
            )
            # out = model.generate(
            #     batch,
            #     max_new_tokens=512,
            #     num_beams=4,
            #     num_return_sequences=1,
            #     output_scores=False,
            #     do_sample=False,
            #     top_p=0.95,
            #     temperature=0.75,
            # )

            for item in out:
                cur = tokenizer.decode(item)
                if "<mol>" not in cur:
                    tmp.append("")
                    continue
                try:
                    cur = (
                        cur.split("<|end_of_text|>")[0]
                        .split("<mol>")[1]
                        .split("</mol>")[0]
                        .replace("<m>", "")
                        .replace("<a>", "")
                        # .replace(" ", "")
                    )
                except Exception as e:
                    print(f"Error: {e} for {cur}")
                cur_smi_set.add(cur)
                tmp.append(cur)
            # if len(cur_smi_set) >= 100:
            #     break
        cur_result.append(tmp)
        print(cur_result)
        result.append(cur_result)
    with open(args.infer_result_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except KeyboardInterrupt:
    #     logger.info("KeyboardInterrupt!")
    # finally:

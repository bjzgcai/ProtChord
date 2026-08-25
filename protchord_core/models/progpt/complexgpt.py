# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
from typing import Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.models.llama import LlamaForCausalLM, LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig

from protchord_core.logging.loggers import logger

# from protchord_core.models.decoder_core.llama2mp_config import MPLlamaConfig
from protchord_core.models.decoder_core.llama_modules import LlamaEmbeddingsPP, LlamaModelPP, NumMLP
from protchord_core.models.progpt.modules.copilotloss import (
    CopilotCriterionsNum,
    CopilotCriterionsNumPP,
    CopilotCriterionsPP,
    CopilotCriterionsWithVAEPP,
)
from protchord_core.models.progpt.modules.encoder_impl import PSMEncoderPP, DualPSMEncoderVAE, ComplexVAEEncoder
from protchord_core.models.encoder.encoder_config import EncoderConfig
from protchord_core.pipeline.accelerator.dataclasses import ModelOutput, TrainStrategy
from protchord_core.pipeline.accelerator.model import Model
from protchord_core.pipeline.accelerator.pipeline_module import SFMPipelineModelMixin
from protchord_core.utils import PretrainedLayerSpec
from protchord_core.utils.optim.optimizer import myAdam, myAdamW
from protchord_core.utils.optim.set_lr import DECAY_COSINE_RATE, groupWarmupDecayLR

from .modules.hybrid_emb import AdaptorConfig, HybridEmbeddings, HybridEmbeddingsPP


def _load_state(path, **kwargs):
    """Load checkpoint tensors with mmap on CPU or directly on the target device."""
    destination = kwargs.get("map_location", "cpu")
    if str(destination) != "cpu" and str(destination) != "cuda" and not isinstance(destination, torch.device):
        destination = "cpu"
    try:
        torch.serialization.add_safe_globals([set])
    except AttributeError:
        pass
    if str(destination) != "cpu":
        return torch.load(path, map_location=destination, weights_only=True)
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except (TypeError, RuntimeError, pickle.UnpicklingError):
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


class LlamaModelLora(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        LORA_R = 8
        LORA_ALPHA = 16
        LORA_DROPOUT = 0.1
        TARGET_MODULES = [
            "q_proj",
            "k_proj",
            "v_proj",
        ]

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
        )

        self.layers = nn.ModuleList(
            [get_peft_model(layer, lora_config) for layer in self.layers]
        )


class LlamaForCausalLMLora(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelLora(config)


# VAE encoder for B1H
# class ComplexVAEEncoder(nn.Module):
#     def __init__(self, hidden_size, latent_size, kl_weight=0.001):
#         super().__init__()
#         self.kl_weight = kl_weight
        
#         # MLPs for mean
#         self.mlp_mu = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.LayerNorm(hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, latent_size)
#         )
        
#         # MLPs for log variance
#         self.mlp_log_var = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.LayerNorm(hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, latent_size)
#         )

#     def masked_pooling(self, x, padding_mask):
#         mask = (~padding_mask).float().unsqueeze(-1)
#         sum_x = (x * mask).sum(dim=1)
#         lengths = mask.sum(dim=1)
#         pooled_x = sum_x / (lengths + 1e-8)
#         return pooled_x.to(dtype=x.dtype)

#     def forward(self, x, padding_mask):
#         x = x.to(dtype=next(self.parameters()).dtype)
#         pooled_x = self.masked_pooling(x, padding_mask)

#         mu = self.mlp_mu(pooled_x)
#         log_var = self.mlp_log_var(pooled_x)

#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         z = mu + eps * std

#         z = z.unsqueeze(1)  # Add sequence dimension

#         # z_projected = self.project_back(z).unsqueeze(1)
#         # noisy_x = x + z_projected

#         return z, mu, log_var

#     # def forward_noise(self, x, noise):
#     #     noisy_x = torch.concat((x, noise), dim=-1)
#     #     noisy_x = self.adaptor_proj(noisy_x)
#     #     return noisy_x

#     def kl_loss(self, mu, log_var):
#         kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1) # size = [bsz]
#         kl_div = kl_div.mean()
#         return kl_div
    
#         # kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
#         # return self.kl_weight * torch.mean(kl_div)


# VAE encoder for BLH
# class ComplexVAEEncoder(nn.Module):
#     def __init__(self, hidden_size, latent_size, kl_weight=0.001):
#         super().__init__()
#         self.kl_weight = kl_weight
        
#         # MLPs for mean
#         self.mlp_mu = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.LayerNorm(hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, latent_size)
#         )
        
#         # MLPs for log variance
#         self.mlp_log_var = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.LayerNorm(hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, latent_size)
#         )

#     def forward(self, x, padding_mask):
#         mu = self.mlp_mu(x)          # (B, L, latent_size)
#         log_var = self.mlp_log_var(x) # (B, L, latent_size)

#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         z = mu + eps * std            # (B, L, latent_size)

#         return z, mu, log_var

#     def kl_loss(self, mu, log_var, padding_mask):
#         kl_div = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
#         mask = (~padding_mask).float().unsqueeze(-1)
#         kl_div = kl_div * mask
#         kl_div = kl_div.sum(dim=[1, 2])
#         kl_div = kl_div.mean()
#         return kl_div


class ComplexGPTModel(Model):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(
        self,
        args,
        vocab_size: int,
        ckp_list: List[str] = [],
    ):
        super().__init__()

        encoder_config = EncoderConfig(args)
        logger.info(f"Trainer args: {args}")
        self.args = args
        llama_config = LlamaConfig.from_pretrained(args.llm_model_name_or_path)
        if llama_config.vocab_size <= 128256:
            llama_config.vocab_size = vocab_size
        # print(f"llama_config.vocab_size: {llama_config.vocab_size}")
        # llama_config.rope_theta = 1000000.0 # *****
        # print("llama rope_theta: 1000000.0")
        args.load_ckpt = args.load_ckpt and not args.ifresume
        # args.load_ckpt = False
        adaptor_config = self.init_adaptor_config(args, llama_config)
        self.num_hidden_layers = llama_config.num_hidden_layers
        self.llama_config = llama_config
        self.use_vae = args.use_vae
        self.add_noise = args.add_noise
        self.vae_beta = args.vae_beta
        self.pipe_layers = []
        self.protein_encoder_type = "encoder"
        if (
            args.strategy != TrainStrategy.ThreeD
            and args.strategy != TrainStrategy.Pipeline
        ):
            self.encoder_impl = PSMEncoderPP(encoder_config)

            if self.args.llm_lora:
                self.decoder = LlamaForCausalLMLora(llama_config)
            else:
                self.decoder = LlamaForCausalLM(llama_config)

            self.decoder.generation_config.pad_token_id = self.decoder.generation_config.eos_token_id
            
            # if self.use_vae:
            self.vae_encoder = ComplexVAEEncoder(
                hidden_size=args.decoder_hidden_dim,
                latent_size=args.vae_latent_size
            )

            self.adaptor = HybridEmbeddingsPP(adaptor_config)
            self.decoder.num_head = NumMLP(
                llama_config.hidden_size, 4 * llama_config.hidden_size, 1
            )
            # self.loss = CopilotCriterionsPP(
            #     self.llama_config, self.llama_config.vocab_size
            # )
            self.loss = CopilotCriterionsWithVAEPP(
                self.llama_config, self.llama_config.vocab_size
            )
        elif args.strategy == TrainStrategy.Pipeline:
            if self.protein_encoder_type != "encoder":
                raise NotImplementedError(
                    "Pipeline strategy currently supports only protein_encoder_type=encoder"
                )
            self.pipe_layers.append(
                PretrainedLayerSpec(
                    # PSMEncoderPP,
                    DualPSMEncoderVAE,
                    encoder_config,
                    load_ckpt=args.load_ckpt,
                    pretrained_ckpt_path=args.encoder_checkpoint_path,
                )
            )

            start_index = 0
            layer1 = _load_state(
                os.path.join(
                    self.args.pretrained_ckpt_path, "layer_00-model_states.pt"
                ),
                map_location=torch.device("cuda"),
            )
            embedding_index = 0
            if "embed_tokens.weight" not in layer1:
                start_index += 2
                embedding_index = 1
            self.pipe_layers.append(
                PretrainedLayerSpec(
                    LlamaEmbeddingsPP,
                    llama_config,
                    new_num_tokens=self.llama_config.vocab_size,
                    load_ckpt=args.load_ckpt,
                    pretrained_ckpt_path=os.path.join(
                        args.pretrained_ckpt_path,
                        "layer_{}-model_states.pt".format(
                            str(embedding_index).zfill(2)
                        ),
                    ),
                    lora_mode="full",
                )
            )

            self.pipe_layers.append(
                PretrainedLayerSpec(
                    HybridEmbeddingsPP,
                    adaptor_config,
                    load_ckpt=args.load_ckpt,
                    pretrained_ckpt_path=args.adapter_ckpt_path,
                )
            )

            self.pipe_layers.extend(
                LlamaModelPP.to_layers(
                    args,
                    llama_config,
                    load_ckpt=args.load_ckpt,
                    new_num_tokens=self.llama_config.vocab_size,
                    start_index=start_index,
                )
            )

            # self.loss = CopilotCriterionsPP(
            #     self.llama_config, self.llama_config.vocab_size
            # )
            self.loss = CopilotCriterionsWithVAEPP(
                self.llama_config, self.llama_config.vocab_size
            )

        else:
            raise NotImplementedError

    def config_optimizer(
        self, model=None
    ) -> Tuple[Optional[Optimizer], Optional[LRScheduler]]:
        # return (None, None)

        if model is None:
            model = self
        if self.args.strategy == TrainStrategy.Pipeline:
            print("#########,optimizer is none!!")
            return (None, None)
        optimizer, _ = myAdam(
            model,
            unfreeze_list=self.args.unfreeze_param_list,
            lr=self.args.max_lr,
            betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.weight_decay,
            eps=1e-8,
        )

        lr_scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=self.args.total_num_steps,
            warmup_max_lr=self.args.max_lr,
            warmup_num_steps=self.args.warmup_num_steps,
            decay_type=DECAY_COSINE_RATE,
        )
        return (optimizer, lr_scheduler)

    def load_ckpt(self):
        checkpoint_cache = {}

        def _load_once(path, **kwargs):
            key = str(path)
            if key not in checkpoint_cache:
                checkpoint_cache[key] = _load_state(path, **kwargs)
            return checkpoint_cache[key]
        if self.protein_encoder_type == "encoder":
            # load psm ckpt
            ckpt_dict = {}
            encoder_impl_model_dict = self.encoder_impl.state_dict()
            layer0 = _load_once(
                self.args.encoder_checkpoint_path, map_location=torch.device("cuda")
            )
            if "module" in layer0:
                layer0 = layer0["module"]
            for k in encoder_impl_model_dict.keys():
                if "net." + k in layer0:
                    new_k = "net." + k
                elif "encoder_impl." + k in layer0:
                    new_k = "encoder_impl." + k
                elif "encoder." + k in layer0:
                    new_k = "encoder." + k
                elif "psm_encoder." + k in layer0:
                    new_k = "psm_encoder." + k  # legacy checkpoint key compatibility
                else:
                    new_k = k
                if new_k not in layer0:
                    continue
                ckpt_dict[k] = layer0[new_k]
            encoder_impl_model_dict.update(ckpt_dict)
            self.encoder_impl.load_state_dict(encoder_impl_model_dict)
            logger.info(f"Loaded path {self.args.encoder_checkpoint_path}")
        else:
            logger.info(
                f"Skip loading PSM checkpoint because protein_encoder_type={self.protein_encoder_type}"
            )

        num_hidden_layers = int(self.num_hidden_layers)
        # load apapter
        if os.path.exists(self.args.adapter_ckpt_path):
            ckpt_dict = {}
            adaptor_model_dict = self.adaptor.state_dict()
            layer1 = _load_once(
                self.args.adapter_ckpt_path, map_location=torch.device("cuda")
            )
            ckpt_dict = {}
            if "module" in layer1:
                layer1 = layer1["module"]
            for k in adaptor_model_dict.keys():
                if "adaptor." + k in layer1:
                    new_k = "adaptor." + k
                else:
                    new_k = k
                ckpt_dict[k] = layer1[new_k]
            adaptor_model_dict.update(ckpt_dict)
            self.adaptor.load_state_dict(adaptor_model_dict)
            logger.info(f"Loaded adaptor in path {self.args.adapter_ckpt_path}")
        else:
            logger.info(f"Adapter path {self.args.adapter_ckpt_path} not exists")

        if os.path.exists(self.args.vae_ckpt_path):
            ckpt_dict = {}
            vae_model_dict = self.vae_encoder.state_dict()
            layer1 = _load_once(
                self.args.vae_ckpt_path, map_location=torch.device("cuda")
            )
            ckpt_dict = {}
            if "module" in layer1:
                layer1 = layer1["module"]
            for k in vae_model_dict.keys():
                if "vae_encoder." + k in layer1:
                    new_k = "vae_encoder." + k
                else:
                    new_k = k
                ckpt_dict[k] = layer1[new_k]
            vae_model_dict.update(ckpt_dict)
            self.vae_encoder.load_state_dict(vae_model_dict)
            logger.info(f"Loaded vae in path {self.args.vae_ckpt_path}")
        else:
            logger.info(f"VAE_encoder path {self.args.vae_ckpt_path} not exists")

        ckpt_dict = {}
        decoder_model_dict = self.decoder.state_dict()

        if os.path.isfile(self.args.pretrained_ckpt_path):
            layer1 = _load_once(
                self.args.pretrained_ckpt_path,
                map_location=torch.device("cuda"),
            )
            if "module" in layer1:
                layer1 = layer1["module"]
            for k in decoder_model_dict.keys():
                if "decoder." + k in layer1:
                    new_k = "decoder." + k
                else:
                    new_k = k
                ckpt_dict[k] = layer1[new_k]
            pass
        else:
            start_index = 0
            layer1 = _load_once(
                os.path.join(
                    self.args.pretrained_ckpt_path, "layer_00-model_states.pt"
                ),
                map_location=torch.device("cuda"),
            )
            if "embed_tokens.weight" in layer1:
                ckpt_dict["model.embed_tokens.weight"] = layer1["embed_tokens.weight"]
            else:
                start_index += 2
                layer1 = _load_once(
                    os.path.join(
                        self.args.pretrained_ckpt_path, "layer_01-model_states.pt"
                    ),
                    map_location=torch.device("cuda"),
                )
                ckpt_dict["model.embed_tokens.weight"] = layer1["embed_tokens.weight"]
            for l in range(0, num_hidden_layers):
                l_index = str(l + 1 + start_index).zfill(2)
                layer = _load_once(
                    os.path.join(
                        self.args.pretrained_ckpt_path,
                        f"layer_{l_index}-model_states.pt",
                    ),
                    map_location=torch.device("cuda"),
                )
                for k in layer:
                    if "dummy" in k or "rotary_emb" in k:
                        continue
                    ckpt_dict[f"model.layers.{l}.{k}"] = layer[k]
                del layer
            layer = _load_once(
                os.path.join(
                    self.args.pretrained_ckpt_path,
                    f"layer_{start_index+num_hidden_layers+1}-model_states.pt",
                ),
                map_location=torch.device("cuda"),
            )
            ckpt_dict["model.norm.weight"] = layer["norm.weight"]

            layer = _load_once(
                os.path.join(
                    self.args.pretrained_ckpt_path,
                    f"layer_{start_index+num_hidden_layers+2}-model_states.pt",
                ),
                map_location=torch.device("cuda"),
            )
            ckpt_dict["lm_head.weight"] = layer["lm_head.weight"]
        decoder_model_dict.update(ckpt_dict)
        self.decoder.load_state_dict(decoder_model_dict)
        logger.info(f"Loaded path {self.args.pretrained_ckpt_path}")

    @torch.no_grad()
    def generate(
        self,
        input_batchdata,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """ """
        # input_tuple, label_tuple = input_batchdata

        if len(input_batchdata) == 2:
            input_tuple, label_tuple = input_batchdata
            # ref_input_tuple = (input_tuple[0], input_tuple[1], input_tuple[2], input_tuple[4])
            if len(input_tuple) > 4:
                ref_input_tuple = (input_tuple[0], input_tuple[1], input_tuple[2], input_tuple[4])
            else:
                ref_input_tuple = (input_tuple[0], input_tuple[1], input_tuple[2], input_tuple[3]) ###
            input_tuple = (input_tuple[0], input_tuple[1], input_tuple[2], input_tuple[3])
        # elif len(input_batchdata) == 3:
        #     input_tuple, ref_psm_batch_data, label_tuple = input_batchdata
        #     ref_input_tuple = (
        #         input_tuple[0],
        #         input_tuple[1],
        #         input_tuple[2],
        #         ref_psm_batch_data,
        #     )

        mol_emb, mol_padding_mask, llm_mask, input_ids = self.encoder_impl(input_tuple)

        if self.use_vae: # conditional
            complex_emb, complex_padding_mask, _, _ = self.encoder_impl(ref_input_tuple)
            noise_z, mu, log_var = self.vae_encoder(
                complex_emb, complex_padding_mask
            )
            prot_mask = (~mol_padding_mask).float().unsqueeze(-1)
            noise_z = noise_z.to(mol_emb.device)
            noise_z = noise_z[:, :mol_emb.size(1), :]       
            mol_emb = mol_emb + noise_z * prot_mask
            # mol_emb = mol_emb + noise_z
        elif self.add_noise:
            bsz, length, hidden_size = mol_emb.shape
            ## B1H
            noise_z = torch.randn(bsz, 1, hidden_size, device=mol_emb.device, dtype=mol_emb.dtype)
            noise_z = noise_z.to(mol_emb.device)
            noise_z = noise_z * self.args.infer_noise_beta
            mol_emb = mol_emb + noise_z

            ## BLH
            # noise_z = torch.randn(bsz, length, hidden_size, device=mol_emb.device, dtype=mol_emb.dtype)
            # noise_z = noise_z.to(mol_emb.device)
            # prot_mask = (~mol_padding_mask).float().unsqueeze(-1)
            # noise_z = noise_z[:, :mol_emb.size(1), :]       
            # mol_emb = mol_emb + noise_z * prot_mask
            
            complex_emb, complex_padding_mask = None, None
            mu, log_var = None, None
        else:
            complex_emb, complex_padding_mask = None, None
            noise_z = None
            mu, log_var = None, None

        psm_batch_data = input_tuple[-1]

        # generate text_emb
        text_embeds = self.decoder.get_input_embeddings()(
            torch.where(input_ids > 0, input_ids, 0)
        )

        if mol_emb is not None:
            (inputs_embeds, llm_mask, position_ids) = self.adaptor(
                (
                    mol_emb, mol_padding_mask, text_embeds, llm_mask, input_ids, 
                    psm_batch_data["pocket_masks"], psm_batch_data["num_atoms"],
                )
            )
            
            # mix embeddings

            outputs = self.decoder.generate(
                inputs_embeds=inputs_embeds,
                **generate_kwargs,
            )

        else:
            # text only generation
            outputs = self.decoder.generate(
                inputs_embeds=text_embeds,
                attention_mask=llm_mask,
                **generate_kwargs,
            )

        return outputs

    def to_layers(self):
        return self.pipe_layers

    def forward(
        self,
        input_batchdata,
    ):
        """ """
        if len(input_batchdata) == 2:
            input_tuple, label_tuple = input_batchdata
            if len(input_tuple) > 4:
                ref_input_tuple = (input_tuple[0], input_tuple[1], input_tuple[2], input_tuple[4])
            else:
                ref_input_tuple = (input_tuple[0], input_tuple[1], input_tuple[2], input_tuple[3]) ###
            input_tuple = (input_tuple[0], input_tuple[1], input_tuple[2], input_tuple[3])
        # elif len(input_batchdata) == 3:
        #     input_tuple, label_tuple, ref_psm_batch_data = input_batchdata
        #     ref_input_tuple = (
        #         input_tuple[0],
        #         input_tuple[1],
        #         input_tuple[2],
        #         ref_psm_batch_data,
        #     )
        with torch.no_grad():
            mol_emb, mol_padding_mask, llm_mask, input_ids = self.encoder_impl(input_tuple)
        # ref_noise
        if self.use_vae:
            with torch.no_grad():
                complex_emb, complex_padding_mask, _, _ = self.encoder_impl(ref_input_tuple)

            # # for fp16
            # dtype = next(self.vae_encoder.parameters()).dtype
            # complex_emb = complex_emb.to(dtype=dtype)
            # with torch.cuda.amp.autocast():
            noise_z, mu, log_var = self.vae_encoder(
                complex_emb, complex_padding_mask
            )

            noise_z = noise_z.to(mol_emb.device)
            mol_emb = mol_emb + noise_z
            kl_loss = self.vae_encoder.kl_loss(mu, log_var)

            # add noise before adaptor according to mol_padding_mask
            # prot_mask = (~mol_padding_mask).float().unsqueeze(-1)
            # noise_z = noise_z.to(mol_emb.device)
            # noise_z = noise_z[:, :mol_emb.size(1), :]       
            # mol_emb = mol_emb + noise_z * prot_mask

            # kl_loss = self.vae_encoder.kl_loss(mu, log_var, complex_padding_mask)
        else:
            complex_emb, complex_padding_mask = None, None
            noise_z = None
            kl_loss = None

        psm_batch_data = input_tuple[-1]

        # generate text_emb
        text_embeds = self.decoder.get_input_embeddings()(
            torch.where(input_ids > 0, input_ids, 0)
        )

        if mol_emb is not None:
            (inputs_embeds, llm_mask, position_ids) = self.adaptor(
                (
                    mol_emb, mol_padding_mask, text_embeds, llm_mask, input_ids, 
                    psm_batch_data["pocket_masks"], psm_batch_data["num_atoms"]
                )
            )

            # mix embeddings

            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=label_tuple[-1].to(inputs_embeds.device),
                return_dict=True,
            )

        else:
            # text only generation
            outputs = self.decoder(
                inputs_embeds=text_embeds,
                attention_mask=label_tuple[-1].to(text_embeds.device),
                return_dict=True,
            )
        logits = outputs["logits"]

        ret = {
            "logits": logits,
            "kl_loss": kl_loss,
        }

        # return logits
        return ret

    def init_adaptor_config(self, args, llama_config):
        config = AdaptorConfig(
            vocab_size=llama_config.vocab_size,
            hidden_size=llama_config.hidden_size,
            intermediate_size=llama_config.intermediate_size,
            num_attention_heads=llama_config.num_attention_heads,
            hidden_act=llama_config.hidden_act,
            rms_norm_eps=llama_config.rms_norm_eps,
            mfm_hidden_size=args.encoder_embed_dim,
            pool_mode=args.pool_mode,
            btn_adaptor=args.btn_adaptor,
            hidden_dropout_prob=args.dropout,
            embedding_length=args.embedding_length,
        )

        return config

    def compute_loss(self, ret, batch) -> ModelOutput:
        # pred is logits of forward output
        # mu, log_var = ret["mu"], ret["log_var"]
        if self.args.strategy != TrainStrategy.Pipeline:
            pred = ret["logits"]
            batch = batch[-1]
            kl_loss = ret["kl_loss"]
            loss, loss_log = self.loss(pred, batch, kl_loss=kl_loss, beta=self.vae_beta)
        if self.args.strategy == TrainStrategy.Pipeline:
            pred = ret[0]
            kl_loss = ret[2] if len(ret)>2 else None
            loss, loss_log = self.loss(pred, batch, kl_loss=kl_loss, beta=self.vae_beta)
        # loss, loss_log = self.loss(pred, batch, mu=mu, log_var=log_var, beta=self.vae_beta)
        
        return ModelOutput(
            loss=loss,
            num_examples=pred.size()[0],
            log_output=loss_log,
        )
    

    @torch.no_grad()
    def extract_prompt_and_ctx(self, batch):
        if len(batch) == 2:
            input_tuple, _ = batch
            ref_input_tuple = (input_tuple[0], input_tuple[1], input_tuple[2], input_tuple[4])
            input_tuple = (input_tuple[0], input_tuple[1], input_tuple[2], input_tuple[3])
        else:
            input_tuple = batch[0]
            ref_input_tuple = (input_tuple[0], input_tuple[1], input_tuple[2], input_tuple[4])

        with torch.no_grad():
            mol_emb, mol_padding_mask, llm_mask, prompt_ids = self.encoder_impl(input_tuple)
        
        if self.add_noise:
            bsz, length, hidden = mol_emb.shape
            noise_z = torch.randn(bsz, 1, hidden, device=mol_emb.device, dtype=mol_emb.dtype) * self.args.infer_noise_beta
            mol_emb = mol_emb + noise_z
            vae_aux = {}
        elif self.use_vae:
            with torch.no_grad():
                complex_emb, complex_padding_mask, _, _ = self.encoder_impl(ref_input_tuple)
            noise_z, mu, log_var = self.vae_encoder(complex_emb, complex_padding_mask)
            noise_z = noise_z.to(mol_emb.device)
            noise_z = noise_z[:, :mol_emb.size(1), :]
            prot_mask = (~mol_padding_mask).float().unsqueeze(-1)
            mol_emb = mol_emb + noise_z * prot_mask
            vae_aux = {"mu": mu, "log_var": log_var}
        else:
            vae_aux = {}

        psm_batch_data = input_tuple[-1]  # pocket_masks / num_atoms 等
        pocket_masks = psm_batch_data["pocket_masks"]
        num_atoms = psm_batch_data["num_atoms"]
        
        ctx = dict(
            mol_emb=mol_emb,
            mol_padding_mask=mol_padding_mask,
            llm_mask=llm_mask,
            # psm_batch_data=psm_batch_data,
            pocket_masks=pocket_masks,
            num_atoms=num_atoms,
            prompt_ids=prompt_ids,
            vae_aux=vae_aux
        )
        return prompt_ids, ctx


    def sequence_logprob_on_ids_with_ctx(self, ctx, input_ids, attention_mask, label_mask):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label_mask = label_mask.to(device)

        mol_emb = ctx["mol_emb"]
        mol_padding = ctx["mol_padding_mask"]
        llm_mask = ctx["llm_mask"]
        # psm_batch_data = ctx["psm_batch_data"]
        pocket_masks = ctx["pocket_masks"]
        num_atoms = ctx["num_atoms"]

        llm_mask = torch.ones_like(input_ids, dtype=torch.bool)
        llm_mask = llm_mask.to(device)

        text_embeds = self.decoder.get_input_embeddings()(
            torch.where(input_ids > 0, input_ids, torch.zeros_like(input_ids))
        )

        inputs_embeds, llm_mask, position_ids = self.adaptor((
            mol_emb, mol_padding, text_embeds, llm_mask, input_ids,
            pocket_masks, num_atoms
        ))

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        logits = outputs.logits.float()
        V = logits.size(-1)

        logits = logits[:, :-1, :]
        target = input_ids[:, 1:].to(torch.long)
        am = attention_mask[:, 1:]
        lm = label_mask[:, 1:]

        valid = (am > 0) & (lm > 0)

        target = target.clone()
        target[~valid] = 0

        target = target.clamp_(min=0, max=V - 1)
        logprobs = torch.log_softmax(logits, dim=-1)
        tok_logp = torch.take_along_dim(
            logprobs, target.unsqueeze(-1), dim=-1
        ).squeeze(-1)

        tok_logp = tok_logp * valid
        ret = tok_logp.sum(dim=-1)
        return ret

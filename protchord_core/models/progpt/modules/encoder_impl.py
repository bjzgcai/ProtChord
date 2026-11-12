# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple

import os
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist

from protchord_core.logging import logger
from protchord_core.models.encoder.modules.timestep_encoder import DiffNoiseEDM
from protchord_core.models.encoder.encoder_config import EncoderConfig
from protchord_core.models.encoder.encoder_model import PSM


@torch.compiler.disable(recursive=True)
def center_pos(batched_data, padding_mask, clean_mask=None):
    # get center of system positions
    is_stable_periodic = batched_data["is_stable_periodic"]  # B x 3 -> B
    periodic_center = (
        torch.gather(
            batched_data["pos"][is_stable_periodic],
            index=batched_data["num_atoms"][is_stable_periodic]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, 3),
            dim=1,
        )
        + torch.gather(
            batched_data["pos"][is_stable_periodic],
            index=batched_data["num_atoms"][is_stable_periodic]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, 3)
            + 7,
            dim=1,
        )
    ) / 2.0
    protein_mask = batched_data["protein_mask"]
    if clean_mask is None:
        num_non_atoms = torch.sum(protein_mask.any(dim=-1), dim=-1)
        non_periodic_center = torch.sum(
            batched_data["pos"].masked_fill(
                padding_mask.unsqueeze(-1) | protein_mask, 0.0
            ),
            dim=1,
        ) / (batched_data["num_atoms"] - num_non_atoms).unsqueeze(-1)
    else:
        # leave out padding tokens when calculating non-atom/non-residue tokens
        num_non_atoms = torch.sum(
            protein_mask.any(dim=-1) | (clean_mask & ~padding_mask), dim=-1
        )
        non_periodic_center = torch.sum(
            batched_data["pos"].masked_fill(
                padding_mask.unsqueeze(-1) | protein_mask | clean_mask.unsqueeze(-1),
                0.0,
            ),
            dim=1,
        ) / (batched_data["num_atoms"] - num_non_atoms).unsqueeze(-1)

    center = non_periodic_center.unsqueeze(1)
    center[is_stable_periodic] = periodic_center
    batched_data["pos"] -= center

    batched_data["pos"] = batched_data["pos"].masked_fill(
        padding_mask.unsqueeze(-1), 0.0
    )
    # TODO: filter nan/inf to zero in coords from pdb data, needs better solution
    batched_data["pos"] = batched_data["pos"].masked_fill(protein_mask, 0.0)
    batched_data["pos"] = batched_data["pos"].masked_fill(
        batched_data["token_id"].eq(156).unsqueeze(-1), 0.0
    )
    return batched_data["pos"]


class PSMEncoderPP(PSM):
    """
    Pipeline parallel mode for PsMEncoder

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        encoder_config: EncoderConfig,
    ) -> None:
        # inheret from PFMEncoder
        args = encoder_config.args
        self.torch_generator = None
        self.encoder_config = encoder_config
        self.sigma_data = encoder_config.edm_sigma_data

        super().__init__(args, encoder_config)

    @classmethod
    def config(cls):
        return cls.encoder_config

    def load_pretrained_weights(self, args, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.

        Args:
            args: Command line arguments.
            checkpoint_path: Path to the pretrained weights.
        """
        checkpoints_state = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoints_state:
            checkpoints_state = checkpoints_state["model"]
        elif "module" in checkpoints_state:
            checkpoints_state = checkpoints_state["module"]

        for key in list(checkpoints_state.keys()):
            if key.startswith("net."):
                checkpoints_state[key[4:]] = checkpoints_state.pop(key)

        IncompatibleKeys = self.load_state_dict(checkpoints_state, strict=False)
        IncompatibleKeys = IncompatibleKeys._asdict()

        missing_keys = []
        for keys in IncompatibleKeys["missing_keys"]:
            if keys.find("dummy") == -1:
                missing_keys.append(keys)

        unexpected_keys = []
        for keys in IncompatibleKeys["unexpected_keys"]:
            if keys.find("dummy") == -1:
                unexpected_keys.append(keys)

        if len(missing_keys) > 0:
            logger.info(
                "Missing keys in {}: {}".format(
                    checkpoint_path,
                    missing_keys,
                )
            )

        if len(unexpected_keys) > 0:
            logger.info(
                "Unexpected keys {}: {}".format(
                    checkpoint_path,
                    unexpected_keys,
                )
            )

        logger.info(f"checkpoint: {checkpoint_path} is loaded")

    def psm_create_system_tags(self, batched_data):
        token_id = batched_data["token_id"]
        sample_type = batched_data["sample_type"]
        is_periodic = batched_data["pbc"].any(dim=-1)
        is_complex = sample_type == 6
        # is_molecule = (~is_periodic) & (token_id <= 129).all(dim=-1)
        is_molecule = (~is_periodic) & (token_id <= 129).all(dim=-1) & (~is_complex)
        is_protein = (~is_periodic.unsqueeze(-1)) & (token_id > 129) & (token_id < 156)
        is_heavy_atom = is_molecule & (token_id > 37).any(dim=-1)
        # is_heavy_atom = is_molecule & (token_id > 130).any(dim=-1)
        
        is_seq_only = sample_type == 5
        is_seq_only = is_seq_only | batched_data["protein_mask"].all(dim=(-1, -2)) # any difference?
        
        is_energy_outlier = is_molecule & (
            torch.abs(batched_data["energy_per_atom"]) > 23
        )
        is_force_outlier = is_molecule & (
            batched_data["forces"].norm(dim=-1).mean(dim=-1) > 2.5
        )

        batched_data["is_periodic"] = is_periodic
        batched_data["is_molecule"] = is_molecule
        batched_data["is_protein"] = is_protein
        batched_data["is_heavy_atom"] = (
            is_energy_outlier | is_heavy_atom | is_force_outlier
        )
        batched_data["is_seq_only"] = is_seq_only
        batched_data["is_complex"] = is_complex

        # atom mask to leave out unit cell corners for periodic systems
        pos = batched_data["pos"]
        n_graphs, n_nodes = pos.size()[:2]

        # create non_atom_mask to mask out unit cell corners for pbc only
        non_atom_mask = torch.arange(
            n_nodes, dtype=torch.long, device=pos.device
        ).unsqueeze(0).repeat(n_graphs, 1) >= batched_data["num_atoms"].unsqueeze(-1)
        batched_data["non_atom_mask"] = non_atom_mask

        # create diff loss mask so that only diffusion loss of 4 out of 8 cell corners are calculated
        diff_loss_mask = torch.arange(
            n_nodes, dtype=torch.long, device=pos.device
        ).unsqueeze(0).repeat(n_graphs, 1) < batched_data["num_atoms"].unsqueeze(-1)
        is_stable_periodic = batched_data["is_stable_periodic"]
        stable_periodic_index = torch.nonzero(is_stable_periodic)[:, 0]
        # print(diff_loss_mask.shape)
        # print(stable_periodic_index.shape)
        # print(is_stable_periodic.shape)
        diff_loss_mask[
            stable_periodic_index, batched_data["num_atoms"][is_stable_periodic]
        ] = True
        diff_loss_mask[
            stable_periodic_index, batched_data["num_atoms"][is_stable_periodic] + 1
        ] = True
        diff_loss_mask[
            stable_periodic_index, batched_data["num_atoms"][is_stable_periodic] + 2
        ] = True
        diff_loss_mask[
            stable_periodic_index, batched_data["num_atoms"][is_stable_periodic] + 4
        ] = True
        batched_data["diff_loss_mask"] = diff_loss_mask

    def psm_create_protein_mask(self, batched_data):
        token_id = batched_data["token_id"]  # B x T
        # create protein aa mask with mask ratio
        batched_data["protein_masked_pos"] = (
            torch.rand_like(token_id.unsqueeze(-1), dtype=torch.float)
            < self.encoder_config.mask_ratio
        ).expand_as(batched_data["pos"])

        # generate a random number [0.15, 0.6] as mask ratio
        if self.encoder_config.mask_ratio < 0.15:
            mask_ratio = self.encoder_config.mask_ratio
        else:
            mask_ratio = np.random.uniform(0.15, self.encoder_config.mask_ratio)

        batched_data["protein_masked_aa"] = (
            torch.rand_like(token_id, dtype=torch.float) < mask_ratio
        )

        masked_pos = batched_data["protein_masked_pos"]

        masked_protein = (
            ((token_id > 129) & (token_id < 158))
            .any(dim=-1, keepdim=True)
            .unsqueeze(-1)
            .expand_as(masked_pos)
        )  # mask_protein: B x T x 3
        masked_nan = (
            torch.isnan(batched_data["pos"])
            .any(dim=-1, keepdim=True)
            .expand_as(masked_pos)
        )  # mask_nan: B x T x 3
        masked_inf = (
            torch.isinf(batched_data["pos"])
            .any(dim=-1, keepdim=True)
            .expand_as(masked_pos)
        )  # mask_nan: B x T x 3

        # protein mask is used to mask out protein inf or nan during training
        mask = masked_protein & (masked_nan | masked_inf)

        # fileter low plddt residues
        # if "confidence" in batched_data:
        #     confidence_mask = (
        #         batched_data["confidence"] < self.encoder_config.plddt_threshold
        #     ) & (batched_data["confidence"] >= 0.0)

        #     mask = mask | confidence_mask.unsqueeze(-1)

        batched_data["protein_mask"] = mask

    def get_continuous_time_step(self, n_graph, device, dtype):
        time_step = torch.zeros(
            [
                n_graph,
            ],
            device=device,
            dtype=dtype,
        )
        time_step = time_step.fill_(
            (self.encoder_config.num_timesteps - 1) * 1.0 / self.encoder_config.num_timesteps
        )
        return time_step

    def psm_edm_sigma(self, rand_normal):
        ## NOTE:There is no sigma_data in EDM but it is used for AF3
        return (
            self.sigma_data
            * (
                rand_normal * self.encoder_config.edm_P_std + self.encoder_config.edm_P_mean
            ).exp()
        )

    def psm_edm_precondition(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = (sigma / self.sigma_data).log() / 4
        return c_skip, c_out, c_in, c_noise

    def forward(self, input_batchdata: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention
        if len(input_batchdata) == 5:
            input_ids, labels, llm_mask, psm_batch_data, ref_psm_batch_data = input_batchdata
        else:
            input_ids, labels, llm_mask, psm_batch_data = input_batchdata
            ref_psm_batch_data = None
        if psm_batch_data["token_id"].shape[1] <= 2:
            return (
                None,
                None,
                llm_mask.cuda(),
                input_ids.cuda(),
            )

        if isinstance(psm_batch_data, dict):
            psm_batch_data = {
                key: value.cuda() if isinstance(value, torch.Tensor) else value
                for key, value in psm_batch_data.items()
            }
        input_ids = input_ids.cuda()
        llm_mask = llm_mask.cuda()
        token_id = psm_batch_data["token_id"]
        self.psm_create_protein_mask(psm_batch_data)

        padding_mask = token_id.eq(0)

        clean_mask = torch.ones_like(token_id, dtype=torch.bool, device=token_id.device)
        clean_mask = clean_mask.masked_fill(token_id == 156, True)
        clean_mask = clean_mask.masked_fill(padding_mask, True)
        clean_mask = clean_mask.masked_fill(
            psm_batch_data["protein_mask"].any(dim=-1), False
        )

        # protien mask set noise and time step start
        if self.torch_generator is None:
            self.torch_generator = torch.Generator(device=psm_batch_data["pos"].device)
            if dist.is_initialized():
                self.torch_generator.manual_seed(dist.get_rank())
        noise = (
            torch.randn(
                psm_batch_data["pos"].size(),
                device=psm_batch_data["pos"].device,
                dtype=psm_batch_data["pos"].dtype,
                generator=self.torch_generator,
            )
            * self.encoder_config.diffusion_noise_std # default=1.0
        )
        psm_batch_data["pos"] = torch.where(
            clean_mask.unsqueeze(-1), psm_batch_data["pos"], noise
        )
        device = psm_batch_data["pos"].device
        n_graph = psm_batch_data["pos"].shape[0]
        # time_step = torch.zeros(
        #     [
        #         n_graph,
        #     ],
        #     device=device,
        #     dtype=psm_batch_data["pos"].dtype,
        # )
        time_step = self.get_continuous_time_step(
            n_graph, device=device, dtype=psm_batch_data["pos"].dtype
        )
        time_step = time_step.unsqueeze(-1).repeat(1, psm_batch_data["pos"].shape[1])
        if clean_mask is not None:
            time_step = time_step.masked_fill(clean_mask, 0.0)

        #  add center pos here
        if self.args.backbone not in ["exp3"]:
            center_pos(psm_batch_data, padding_mask)

        psm_batch_data["pos"] = torch.where(
            clean_mask.unsqueeze(-1), psm_batch_data["pos"], noise
        )
        # protien mask set noise and time step end

        token_id = psm_batch_data["token_id"]
        padding_mask = token_id.eq(0)
        self.psm_create_system_tags(psm_batch_data)
        # import ipdb; ipdb.set_trace()
        # import pickle

        noise_step = None
        if self.encoder_config.diffusion_mode == "edm":
            nnodes = psm_batch_data["pos"].shape[1]
            noise_step = torch.randn(
                (n_graph,), dtype=psm_batch_data["pos"].dtype, device=device
            )
            noise_step = noise_step.unsqueeze(-1).repeat(1, nnodes)
            noise_step = noise_step.masked_fill(token_id == 156, -4.42)

            # set T noise if protein is seq only
            # set 0 noise for padding
            noise_step = noise_step.masked_fill(padding_mask, -4.42)
            # # TODO: found this may cause instability issue, need to check
            # # # set T noise for batched_data["protein_mask"] nan/inf coords
            noise_step = noise_step.masked_fill(
                psm_batch_data["protein_mask"].any(dim=-1), 4.19
            )

            # noise_step = noise_step.unsqueeze(-1).repeat(1, nnodes)
            noise_step = noise_step.masked_fill(clean_mask, -4.42)
            t_hat = self.psm_edm_sigma(time_step).unsqueeze(-1)

            # Reshape sigma to (B, L, 1)
            # print(t_hat.shape)
            # t_hat = (
            #     t_hat.unsqueeze(-1)
            #     .repeat(1, psm_batch_data["pos"].shape[1])
            #     .unsqueeze(-1)
            # )
            t_hat = t_hat.double()
            t_hat = t_hat.masked_fill(clean_mask.unsqueeze(-1), 0.0064)

            c_skip, c_out, c_in, c_noise = self.psm_edm_precondition(t_hat)
            psm_batch_data["edm_sigma"] = t_hat
            psm_batch_data["c_skip"] = c_skip
            psm_batch_data["c_out"] = c_out
            psm_batch_data["c_in"] = c_in
            psm_batch_data["c_noise"] = c_noise
            time_step = None

        if token_id.shape[1] > 2:
            result_dict = super().forward(
                psm_batch_data,
                time_step=time_step, # None
                clean_mask=clean_mask,
                padding_mask=padding_mask,
                noise_step=noise_step,
            )
            x = result_dict["decoder_x_output"]
        else:
            x = None
            padding_mask = None
        # exit(0)
        # print('mean and std')
        # print(f"psm mean{torch.mean(x).item()}")
        # print(f"psm std{torch.std(x).item()}")
        return (
            x,
            padding_mask,
            llm_mask,
            input_ids,
            # psm_batch_data['pocket_masks'],
            # psm_batch_data['num_atoms'],
        )


class ComplexVAEEncoder(nn.Module):
    def __init__(self, hidden_size, latent_size, kl_weight=0.001):
        super().__init__()
        self.kl_weight = kl_weight
        
        # MLPs for mean
        self.mlp_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )
        
        # MLPs for log variance
        self.mlp_log_var = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )

    def masked_pooling(self, x, padding_mask):
        mask = (~padding_mask).float().unsqueeze(-1)
        sum_x = (x * mask).sum(dim=1)
        lengths = mask.sum(dim=1)
        pooled_x = sum_x / (lengths + 1e-8)
        return pooled_x.to(dtype=x.dtype)

    def forward(self, x, padding_mask):
        x = x.to(dtype=next(self.parameters()).dtype)
        pooled_x = self.masked_pooling(x, padding_mask)

        mu = self.mlp_mu(pooled_x)
        log_var = self.mlp_log_var(pooled_x)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        z = z.unsqueeze(1)  # Add sequence dimension

        # z_projected = self.project_back(z).unsqueeze(1)
        # noisy_x = x + z_projected

        return z, mu, log_var

    # def forward_noise(self, x, noise):
    #     noisy_x = torch.concat((x, noise), dim=-1)
    #     noisy_x = self.adaptor_proj(noisy_x)
    #     return noisy_x

    def kl_loss(self, mu, log_var):
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1) # size = [bsz]
        kl_div = kl_div.mean()
        return kl_div
    
        # kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        # return self.kl_weight * torch.mean(kl_div)


class DualPSMEncoderVAE(nn.Module):
    def __init__(
        self,
        encoder_config: EncoderConfig,
    ) -> None:
        super().__init__()
        args = encoder_config.args
        self.torch_generator = None
        self.args = args
        self.use_vae = args.use_vae
        self.encoder_config = encoder_config
        self.encoder_impl = PSMEncoderPP(encoder_config)
        self.vae_encoder = ComplexVAEEncoder(
            hidden_size=args.decoder_hidden_dim,
            latent_size=args.vae_latent_size,
            kl_weight=args.vae_kl_weight,
        )

    def forward(self, input_batchdata: Tuple):
        input_ids, labels, input_ids_masks, psm_batch_data, ref_psm_batch_data = input_batchdata
        input_tuple = (
            input_ids,
            labels,
            input_ids_masks,
            psm_batch_data
        )
        with torch.no_grad():
            mol_emb, mol_padding_mask, llm_mask, input_ids = self.encoder_impl(input_tuple)

        if self.use_vae:
            ref_input_tuple = (
                input_ids,
                labels,
                input_ids_masks,
                ref_psm_batch_data
            )
            with torch.no_grad():
                complex_emb, complex_padding_mask, _, _ = self.encoder_impl(ref_input_tuple)
            noise_z, mu, log_var = self.vae_encoder(
                complex_emb, complex_padding_mask
            )
            noise_z = noise_z.to(mol_emb.device)
            mol_emb = mol_emb + noise_z
            kl_loss = self.vae_encoder.kl_loss(mu, log_var)
        else:
            kl_loss = None

        if kl_loss is None:
            ret = (
                mol_emb,
                mol_padding_mask,
                llm_mask,
                input_ids,
                psm_batch_data['pocket_masks'],
                psm_batch_data['num_atoms'],
            )
        else:
            ret = (
                mol_emb,
                mol_padding_mask,
                llm_mask,
                input_ids,
                psm_batch_data['pocket_masks'],
                psm_batch_data['num_atoms'],
                kl_loss,
            )
        # print("-------------------------- PSMEncoder is OK!")
        return ret

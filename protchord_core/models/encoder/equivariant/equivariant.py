# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Optional

import torch
import torch.nn as nn

from protchord_core.models.encoder.encoder_config import EncoderConfig

from ..modules.pbc import CellExpander
from .geomformer import GeomFormer


class EquivariantEncoder(nn.Module):
    def __init__(self):
        super(EquivariantEncoder, self).__init__()

    def forward(self, x):
        pass


class EquivariantDecoder(nn.Module):
    def __init__(self, encoder_config: EncoderConfig):
        super(EquivariantDecoder, self).__init__()
        self.encoder_config = encoder_config

        # use GeoMFormer as equivariant decoder
        self.model = GeomFormer(
            encoder_config=encoder_config,
            num_pred_attn_layer=encoder_config.num_pred_attn_layer,
            embedding_dim=encoder_config.embedding_dim,
            num_attention_heads=encoder_config.num_attention_heads,
            ffn_embedding_dim=encoder_config.decoder_ffn_dim,
            dropout=encoder_config.dropout,
            attention_dropout=encoder_config.attention_dropout,
            activation_dropout=encoder_config.activation_dropout,
            num_3d_bias_kernel=encoder_config.num_3d_bias_kernel,
            num_edges=encoder_config.num_edges,
            num_atoms=encoder_config.num_atoms,
        )

        self.cell_expander = CellExpander(
            encoder_config.pbc_expanded_distance_cutoff,
            encoder_config.pbc_expanded_token_cutoff,
            encoder_config.pbc_expanded_num_cell_per_direction,
            encoder_config.pbc_multigraph_cutoff,
        )

    def forward(
        self,
        batched_data,
        x,
        mixed_attn_bias,
        padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
        time_embed: Optional[torch.Tensor] = None,
    ):
        pos = batched_data["pos"]
        return self.model(
            batched_data,
            x,
            pos,
            mixed_attn_bias,
            padding_mask,
            pbc_expand_batched,
            time_embed=time_embed,
        )

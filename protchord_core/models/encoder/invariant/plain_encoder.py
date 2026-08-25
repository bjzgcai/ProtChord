# -*- coding: utf-8 -*-
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from protchord_core.models.encoder.invariant.plain_encoder_layer import (
    PSMPairPlainEncoderLayer,
    PSMPlainEncoderLayer,
)
from protchord_core.models.encoder.encoder_config import EncoderConfig
from protchord_core.modules.droppath import DropPath
from protchord_core.modules.FairseqDropout import FairseqDropout

# from fairseq import utils
from protchord_core.modules.get_activation_fn import get_activation_fn
from protchord_core.modules.mem_eff_attn import MemEffSelfAttn


class PSMPlainEncoder(nn.Module):
    """
    Implements a Transformer-M Encoder Layer.
    """

    def __init__(self, args, encoder_config: EncoderConfig):
        super().__init__()

        self.layers = nn.ModuleList([])

        for nl in range(encoder_config.num_encoder_layers):
            self.layers.extend([PSMPlainEncoderLayer(args, encoder_config)])

        # dummy param for lora, do not remove
        self.dummy = nn.Linear(1, 1, bias=False)

    @torch.compiler.disable(recursive=False)
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        batched_data: Dict,
        pbc_expand_batched: Optional[Dict] = None,
        mixed_attn_bias: Optional[torch.Tensor] = None,
        ifbackprop: bool = False,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        for layer_index, layer in enumerate(self.layers):
            x, _ = layer(
                x,
                padding_mask,
                batched_data,
                mixed_attn_bias=mixed_attn_bias,
                pbc_expand_batched=pbc_expand_batched,
                ifbackprop=ifbackprop,
            )
        return x


class PSMPairPlainEncoder(nn.Module):
    """
    Implements a Transformer-M Encoder Layer.
    """

    def __init__(self, args, encoder_config: EncoderConfig):
        super().__init__()

        self.layers = nn.ModuleList([])

        for nl in range(encoder_config.num_encoder_layers):
            self.layers.extend([PSMPairPlainEncoderLayer(args, encoder_config)])

        # dummy param for lora, do not remove
        self.ln_pair = nn.LayerNorm(encoder_config.encoder_pair_embed_dim)
        self.dummy = nn.Linear(1, 1, bias=False)

    @torch.compiler.disable(recursive=False)
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        batched_data: Dict,
        pbc_expand_batched: Optional[Dict] = None,
        mixed_attn_bias: Optional[torch.Tensor] = None,
        ifbackprop: bool = False,
        x_pair: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        for _, layer in enumerate(self.layers):
            x, x_pair = layer(
                x,
                padding_mask,
                batched_data,
                mixed_attn_bias=mixed_attn_bias,
                x_pair=x_pair,
                pbc_expand_batched=pbc_expand_batched,
                ifbackprop=ifbackprop,
            )

        x_pair = self.ln_pair(x_pair)

        return x, x_pair.permute(2, 0, 1, 3)

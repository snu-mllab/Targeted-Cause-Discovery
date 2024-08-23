# Codes are adapted from: https://github.com/rmwu/sea (Author: Menghua Wu)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialTransformer(nn.Module):
    """ 2D row/col attention over features (n, m, d) 
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        # data value
        self.embed_value = nn.Linear(1, args.embed_dim)
        # intervention binary
        self.embed_intv = nn.Embedding(2, args.embed_dim)
        # N_node (use padding_idx)
        self.projection = nn.Linear(2 * args.embed_dim, args.embed_dim)
        self.emb_layer_norm_before = ESM1bLayerNorm(2 * self.args.embed_dim)

        self.dropout_module = nn.Dropout(self.args.dropout)

        # Transformer encoder layers
        layers = []
        for _ in range(self.args.transformer_num_layers):
            layer = AxialTransformerLayer(
                embed_dim=self.args.embed_dim,
                ffn_embed_dim=self.args.ffn_embed_dim,
                n_heads=self.args.n_heads,
                dropout=self.args.dropout,
                max_tokens=1e8,
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)

    def forward(self, batch, repr_layers=[]):
        """
            data = batch_size (B), num_trials (R), num_nodes (C)
            intv = batch_size, num_trials, num_nodes
        """
        # expand inputs
        data = batch["data"].float()

        data_embed = self.embed_value(data.unsqueeze(-1))
        intv_embed = self.embed_intv(batch["intv"].int())
        x = torch.cat([data_embed, intv_embed], dim=-1)

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)

        x = self.emb_layer_norm_before(x)
        x = self.dropout_module(x)
        x = self.projection(x)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, padding_mask=None)
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.permute(2, 0, 1, 3)

        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D

        return x

    @property
    def num_layers(self):
        return self.args.transformer_num_layers


class AxialTransformerLayer(nn.Module):
    """
        Single block
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        n_heads: int,
        dropout: float,
        max_tokens: int,
        ffn_embed_dim: int,
        scale_rows=False,
        scale_cols=False,
        activation_dropout: float = 0.1,
    ):
        super().__init__()
        """
            @param scale_rows  True to scale rows, for invariance to # cols
            @param scale_cols  True to scale cols, for invariance to # rows
        """
        # Save params
        self.embed_dim = embed_dim
        self.dropout = dropout
        # 2D attention over rows and columns
        # Shared arguments for all attention / FFN layers
        attn_kwargs = {
            "embed_dim": embed_dim,
            "num_heads": n_heads,
            "dropout": dropout,
            "max_tokens": max_tokens,
        }
        row_attn = SelfAttention2D(dim=0, use_scaling=scale_rows, **attn_kwargs)
        col_attn = SelfAttention2D(dim=1, use_scaling=scale_cols, **attn_kwargs)
        ffn = FeedForwardNetwork(embed_dim, ffn_embed_dim, activation_dropout=dropout)
        # Add residual wrapper
        self.row_attn = self.build_residual(row_attn)
        self.col_attn = self.build_residual(col_attn)
        self.ffn = self.build_residual(ffn)

    def forward(self, x, padding_mask=None):
        """
            x = batch_size, num_rows, num_cols, embed_dim
        """
        x = self.row_attn(x, padding_mask=padding_mask)
        x = self.col_attn(x, padding_mask=padding_mask)
        x = self.ffn(x)
        return x

    def build_residual(self, layer: nn.Module):
        """
            Wrap layer with LayerNorm and residual
        """
        return NormalizedResidualBlock(layer, self.embed_dim, self.dropout)


class SelfAttention2D(nn.Module):
    """
        Heavily modified from:
        https://github.com/facebookresearch/esm/blob/main/esm/model/msa_transformer.py

        Compute self-attention over rows of a 2D input.
        This module's correctness was tested in src/model/attn_test.py
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        max_tokens: int = 2**16,
        dim=0,
        use_scaling=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.max_tokens = max_tokens
        assert dim in [0, 1], f"dim {dim} not allowed; 2D inputs only, [0, 1]"
        self.dim = dim
        self.use_scaling = use_scaling
        self.attn_shape = "hnij"

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def align_scaling(self, q):
        if not self.use_scaling:
            return 1.0
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    def _batched_forward(
        self,
        x,
        padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_rows = max(1, self.max_tokens // num_cols)
        outputs = []
        for start in range(0, num_rows, max_rows):
            output = self(
                x[start:start + max_rows],
                padding_mask=padding_mask[:, start:start +
                                          max_rows] if padding_mask is not None else None,
            )
            outputs.append(output)
        return outputs

    def compute_attention_weights(
        self,
        x,
        scaling: float,
        padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        q *= scaling
        if padding_mask is not None:
            # Zero out any padded aligned positions - this is important since
            # we take a sum across the alignment axis.
            q *= 1 - padding_mask.permute(1, 2, 0)[..., None, None].to(q)

        # r = row, i = index into col, n = batch_size, h = heads, d = head_dim
        attn_weights = torch.einsum(f"rinhd,rjnhd->{self.attn_shape}", q, k)

        if padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                padding_mask[:, 0][None, :, None],
                -10000,
            )

        return attn_weights

    def compute_attention_update(
        self,
        x,
        padding_mask=None,
    ):
        """
            x: [R, C, B, d]
            padding_mask: [B, R, C]
        """
        num_rows, num_cols, batch_size, embed_dim = x.size()
        # compute attention weights
        scaling = self.align_scaling(x)
        attn_weights = self.compute_attention_weights(x, scaling, padding_mask)
        attn_probs = attn_weights.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)
        # apply update
        v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        context = torch.einsum(f"{self.attn_shape},rjnhd->rinhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        return output

    def forward(
        self,
        x,
        padding_mask=None,
    ):
        """
            x: [R, C, B, d]
            padding_mask: [B, R, C]  DIFFERENT from x order ???
        """
        # permute
        if self.dim == 1:
            x = x.permute(1, 0, 2, 3)
            if padding_mask is not None:
                padding_mask = padding_mask.transpose(1, 2)
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if (num_rows * num_cols > self.max_tokens) and not torch.is_grad_enabled():
            output = self._batched_forward(x, padding_mask)
            output = torch.cat(output, dim=0)
        else:
            output = self.compute_attention_update(x, padding_mask)
        # permute back
        if self.dim == 1:
            output = output.permute(1, 0, 2, 3)
        return output


class NormalizedResidualBlock(nn.Module):
    """
        This class is unchanged
    """

    def __init__(
        self,
        layer: nn.Module,
        embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.layer = layer
        self.dropout_module = nn.Dropout(dropout,)
        self.layer_norm = ESM1bLayerNorm(self.embed_dim)

    def forward(self, x, *args, **kwargs):
        residual = x
        x = self.layer_norm(x)
        outputs = self.layer(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x = outputs
            out = None

        x = self.dropout_module(x)
        x = residual + x

        if out is not None:
            return (x,) + tuple(out)
        else:
            return x


class FeedForwardNetwork(nn.Module):
    """
        This class is unchanged
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        activation_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(activation_dropout,)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x


class TopLayer(nn.Module):

    def __init__(self, args, output_dim):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.linear = nn.Linear(args.embed_dim, args.ffn_embed_dim)
        self.layer_norm = ESM1bLayerNorm(args.ffn_embed_dim)
        self.linear2 = nn.Linear(args.ffn_embed_dim, output_dim)

        self.args = args

    def forward(self, features):
        # B x R x C x D -> B x C x D
        x = features.mean(dim=1)
        x = self.linear(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.linear2(x)  # B x C x D
        return x


class TopLayer_Perm(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.head1 = TopLayer(args, args.output_dim)
        self.head2 = TopLayer(args, args.output_dim)

    def forward(self, features):
        u = self.head1(features)
        v = self.head2(features)

        x = torch.matmul(u, v.transpose(-2, -1))
        return x


try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):

        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm

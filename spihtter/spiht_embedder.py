"""
This file provides a neural network, which provides embeddings that provide
context about the internal state of the Spiht algorithm. The input to this
network is a matrix of Spiht metadata tokens. This metadata matrix can be
obtained using the spiht streaming decoder, or the spiht.decode_image function.
"""
import math
import torch
from torch import nn


def unpackbits_pt(x, num_bits):
    xshape = x.shape
    x = x.reshape(-1, 1)
    mask = 2 ** torch.arange(num_bits, dtype=x.dtype, device=x.device).reshape(
        1, num_bits
    )
    return (x & mask).to(torch.bool).reshape(*xshape, num_bits)


class CAPE2d(nn.Module):
    """
    https://arxiv.org/abs/2106.03143
    2021
    CAPE: Encoding Relative Positions with Continuous Augmented Positional Embeddings

    some code from https://github.com/gcambara/cape
    """

    def __init__(
        self,
        d_model: int,
        max_global_shift: float = 0.01,
        max_local_shift: float = 0.0,
        max_global_scaling: float = 1.0,
    ):
        super().__init__()

        assert (
            d_model % 2 == 0
        ), f"""The number of channels should be even,
                                     but it is odd! # channels = {d_model}."""

        self.max_global_shift = max_global_shift
        self.max_local_shift = max_local_shift
        self.max_global_scaling = max_global_scaling

        half_channels = d_model // 2
        rho = 10 ** torch.linspace(0, 1, half_channels)
        w_x = rho * torch.cos(torch.arange(half_channels))
        w_y = rho * torch.sin(torch.arange(half_channels))
        self.register_buffer("w_x", w_x)
        self.register_buffer("w_y", w_y)

    def forward(self, x, y):
        return self.compute_pos_emb(x, y)

    def compute_pos_emb(self, x, y):
        """
        x: batched tensor of relative x positions, from -1 to 1
        y: batched tensor of relative y positions, from -1 to 1
        """
        x, y = self.augment_positions(x, y)

        phase = torch.pi * (self.w_x * x.unsqueeze(-1) + self.w_y * y.unsqueeze(-1))
        pos_emb = torch.concatenate([torch.cos(phase), torch.sin(phase)], axis=-1)

        return pos_emb

    def augment_positions(self, x, y):
        if self.training:
            if self.max_global_shift:
                x = x + torch.empty_like(x).uniform_(
                    -self.max_global_shift, self.max_global_shift
                )

                y = y + torch.empty_like(y).uniform_(
                    -self.max_global_shift, self.max_global_shift
                )

            if self.max_local_shift:
                raise NotImplementedError()

            if self.max_global_scaling > 1.0:
                log_l = math.log(self.max_global_scaling)
                lambdas = torch.exp(torch.empty_like(x).uniform_(-log_l, log_l))
                x *= lambdas
                y *= lambdas

        return x, y


class SpihtEmbedder(nn.Module):
    def __init__(
        self,
        dim=512,
        action_size=8,
        max_dwt_depth=12,
        dwt_channels=3,
    ):
        """
        Args:
            dim: dimension of input embeddings
            encoder/decoder acts over bits, so this value is set to 4 by
            default, for the 2 bits and then the start token and end token.
            patch_size: Spiht bits are combined into input tokens of this
            length. By default this is 8, which corresponds to bytes.
            action_size: The Spiht algorithm has 6 distinct places in the algorithm where bits are inputted/outputted.
            max_dwt_depth: This is the maximum depth, (number of levels), of the DWT coefficients.
            max_dwt_h: Maximum height of the coefficients in a filter array, the max image height is double this.
            max_dwt_w: Maximum height of the coefficients in a filter array, the max image height is double this.
            dwt_channels: Number of channels of DWT coeffs
        """
        super().__init__()

        self.dim = dim
        self.action_size = action_size
        self.max_dwt_depth = max_dwt_depth
        self.dwt_channels = dwt_channels

        emb_dim = dim

        # TODO uses simple absolute pos embeddings
        self.pos_emb = CAPE2d(emb_dim)
        self.pos_scale = nn.Parameter(torch.Tensor([1 / self.dim**0.5]))

        self.dwt_depth_embed = nn.Embedding(max_dwt_depth, emb_dim)
        self.dwt_channel_embed = nn.Embedding(dwt_channels, emb_dim)
        # ll, da, ad, 'dd'
        self.dwt_filter_embed = nn.Embedding(4, emb_dim)
        self.action_embed = nn.Embedding(action_size, emb_dim)

        self.n_emb = nn.Embedding(2**4, emb_dim)

        # input features are 16 bits of the input int16
        self.rec_arr_proj = nn.Sequential(
            nn.Linear(16, emb_dim, bias=False),
        )

        self.pad_token = nn.Embedding(1, dim)

    def forward(
        self,
        metadata_ids: torch.LongTensor = None,
    ):
        """
        metadata_ids: These are ids of shape B,S,7
            The last dimension is a feature vector describing the
            intermediate details of the spiht algorithm for the NEXT
            token to be decoded.

            This vector has in the following order:
                action_ids: The type of action for the NEXT bit.
                    This is in [0-6]
                height_ids: The relative height of the NEXT coefficient that has a
                    detail being output, in the range -100000, 100000
                width_ids: The width of the NEXT coefficient that has a
                    detail being output, in the range -100000, 100000
                channel_ids: The channel of the NEXT coefficient
                filter_ids: The ID of the NEXT filter of the coefficient, which
                is in [0, 3]
                depth_ids: The depth of the NEXT coefficient
                n_ids: The SPIHT variable 'n' of the NEXT coefficient
                rec_arr_values: The value of the NEXT coefficient
        Returns:
            tensor of shape B, T, Z
            where Z is self.dim
        """

        # if all ids are 0 then will pad
        # TODO have explicit pad ID
        pad_mask = (metadata_ids == 0).all(-1)

        # unpacks
        (
            action_ids,
            height_ids,
            width_ids,
            channel_ids,
            filter_ids,
            depth_ids,
            n_ids,
            rec_arr_values,
        ) = metadata_ids.movedim(-1, 0)

        action_emb = self.action_embed(action_ids)

        # scales to -1, 1
        x_pos, y_pos = height_ids / 100_000, width_ids / 100_000
        pos_embed = self.pos_emb(x_pos, y_pos) * self.pos_scale

        channel_emb = self.dwt_channel_embed(channel_ids)
        filter_emb = self.dwt_filter_embed(filter_ids)
        depth_emb = self.dwt_depth_embed(depth_ids)

        n_emb = self.n_emb(n_ids)

        # converts rec_arr_values into bit labels
        # each int gets 16 bits
        # all values are assumed to be between -2**15 and 2**15
        rec_arr_values = rec_arr_values + 2**15
        bits = unpackbits_pt(rec_arr_values, 16).to(self.rec_arr_proj[0].weight.dtype)
        rec_arr_values_emb = self.rec_arr_proj(bits)

        # adds them all up
        # There could be better ways of combining so much conditioning information
        embed = (
            action_emb
            + channel_emb
            + filter_emb
            + depth_emb
            + n_emb
            + pos_embed
            + rec_arr_values_emb
        )

        embed = embed * (~pad_mask).unsqueeze(-1)

        embed = embed + pad_mask.unsqueeze(-1) * self.pad_token.weight[0]

        return embed

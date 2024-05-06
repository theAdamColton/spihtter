"""
This file provides a neural network, which provides embeddings that provide
context about the internal state of the Spiht algorithm. The input to this
network is a matrix of Spiht metadata tokens. This metadata matrix can be
obtained using the spiht streaming decoder, or the spiht.decode_image function.
"""

import torch
from torch import nn


def unpackbits_pt(x, num_bits, mask=None):
    xshape = x.shape
    x = x.view(-1, 1)
    if mask is not None:
        mask = 2 ** torch.arange(num_bits, dtype=x.dtype, device=x.device).unsqueeze(0)
    return (x & mask).to(torch.bool).view(*xshape, num_bits)


class SpihtEmbedder(nn.Module):
    def __init__(
        self,
        dim=512,
        action_size=8,
        max_dwt_depth=12,
        dwt_channels=3,
        max_height=128,
        max_width=128,
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

        self.pos_embed_height = nn.Embedding(max_height, dim)
        self.pos_embed_width = nn.Embedding(max_width, dim)

        self.dwt_depth_embed = nn.Embedding(max_dwt_depth, dim)
        self.dwt_channel_embed = nn.Embedding(dwt_channels, dim)

        # ll, da, ad, 'dd'
        self.dwt_filter_embed = nn.Embedding(4, dim)

        self.action_embed = nn.Embedding(action_size, dim)

        self.n_emb = nn.Embedding(2**4, dim)

        # input features are 16 bits of the input int16
        self.rec_arr_proj = nn.Sequential(
            nn.Linear(16, dim, bias=False),
        )

        self.pad_token = nn.Embedding(1, dim)

        self.rec_arr_num_bits = 16

        self.register_buffer(
            "unpack_bits_mask",
            2 ** torch.arange(self.rec_arr_num_bits, dtype=torch.long).unsqueeze(0),
        )

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
        ) = metadata_ids.unbind(-1)

        action_emb = self.action_embed(action_ids)

        pos_embed = self.pos_embed_height(height_ids) + self.pos_embed_width(width_ids)

        channel_emb = self.dwt_channel_embed(channel_ids)
        filter_emb = self.dwt_filter_embed(filter_ids)
        depth_emb = self.dwt_depth_embed(depth_ids)

        n_emb = self.n_emb(n_ids)

        # converts rec_arr_values into bit labels
        # each int gets 16 bits
        # all values are assumed to be between -2**15 and 2**15
        rec_arr_values = rec_arr_values + 2**15
        # the bits of rec_arr_values, as 1. or -1.
        bits = (
            unpackbits_pt(rec_arr_values, 16, mask=self.unpack_bits_mask).to(
                self.rec_arr_proj[0].weight.dtype
            )
            * 2
            - 1
        )
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

        embed = (
            embed * (~pad_mask).unsqueeze(-1)
            + pad_mask.unsqueeze(-1) * self.pad_token.weight[0]
        )

        return embed

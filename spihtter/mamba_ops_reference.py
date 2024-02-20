"""
fallback native python/pytorch reference implementation of the mamba ops

see https://github.com/Dao-AILab/causal-conv1d
and https://github.com/state-spaces/mamba/blob/009bec5ee37f586844a3fc89c040a9c1a9d8badf/mamba_ssm/ops/selective_scan_interface.py
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def causal_conv1d_ref(x, weight, bias=None, activation=None):
    b, z, s = x.shape
    z, width = weight.shape

    x = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=z)[..., :s]

    if activation == "silu":
        return F.silu(x)
    else:
        raise ValueError(activation)


def selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(
                rearrange(B.float(), "... (L two) -> ... L two", two=2)
            )
        if is_variable_C:
            C = torch.view_as_complex(
                rearrange(C.float(), "... (L two) -> ... L two", two=2)
            )
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            else:
                y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def selective_scan_py_fast(us, dts, As, Bs, Cs, Ds, z = None, delta_bias=None, delta_softplus=False, chunksize=64, eps=1e-11):
    """
    TODO return last state
    Code from:
    https://github.com/MzeroMiko/mamba-mini/blob/8935646bfc40d4a4b9ef855109d3e16a917917ad/test_selective_scan.py


    # B: batch_size, G: groups, D: dim, N: state dim, L: seqlen
    us: B, G * D, L 
    dts: B, G * D, L
    As: G * D, N
    Bs: B, G, N, L
    Cs: B, G, N, L
    Ds: G * D
    delta_bias: G * D
    # chunksize can be any as you like. But as the chunksize raises, hs may get None, as exp(sum(delta) A) is really small
    """
    def selective_scan_chunk(us, dts, As, Bs, Cs, Ds, hprefix):
        """
        partial(h) / partial(t) = Ah + Bu; y = Ch + Du;
        => partial(h*exp(-At)) / partial(t) = Bu*exp(-At);
        => h_t = h_0 + sum_{0}_{t}_{Bu*exp(A(t-v)) dv};
        => h_b = exp(A(dt_a + ... + dt_{b-1})) * (h_a + sum_{a}_{b-1}_{Bu*exp(-A(dt_a + ... + dt_i)) dt_i});
           y_i = C_i*h_i + D*u_i
        """
        """
        us: (L, B, G, D) # L is chunk_size
        dts: (L, B, G, D)
        As: (G, D, N)
        Bs: (L, B, G, N)
        Cs: (L, B, G, N)
        Ds: (G, D)
        hprefix: (1, B, G, D, N)
        """
        
        Ats = torch.einsum("gdn,lbgd->lbgdn", As, dts.cumsum(dim=0)).exp()
        dtBus = torch.einsum("lbgd,lbgn,lbgd->lbgdn", dts, Bs, us)
        hs = Ats * (dtBus / (Ats + eps)).cumsum(dim=0) + Ats * hprefix.unsqueeze(0)
        ys = torch.einsum("lbgn,lbgdn->lbgd", Cs, hs) 
        if Ds is not None:
            ys = ys + Ds * us
        return ys, hs[-1]
    
    dts = dts.float()
    if delta_bias is not None:
        dts = dts + delta_bias.view(1, -1, 1).float()
    if delta_softplus:
        dts = torch.nn.functional.softplus(dts)
    
    if len(Bs.shape) == 3:
        Bs = Bs.unsqueeze(1)
    if len(Cs.shape) == 3:
        Cs = Cs.unsqueeze(1)
    B, G, N, L = Bs.shape
    us = us.view(B, G, -1, L).permute(3, 0, 1, 2).float()
    dts = dts.view(B, G, -1, L).permute(3, 0, 1, 2).float()
    As = As.view(G, -1, N).float()
    Bs = Bs.permute(3, 0, 1, 2).float()
    Cs = Cs.permute(3, 0, 1, 2).float()
    Ds = Ds.view(G, -1).float() if Ds is not None else None
    D = As.shape[1]
    
    oys = []
    hprefix = us.new_zeros((B, G, D, N), dtype=torch.float)
    for i in range(0, L, chunksize):
        ys, hprefix = selective_scan_chunk(
            us[i:i + chunksize], dts[i:i + chunksize], 
            As, Bs[i:i + chunksize], Cs[i:i + chunksize], Ds, hprefix, 
        )
        oys.append(ys)

    oys = torch.cat(oys, dim=0).permute(1, 2, 3, 0).view(B, -1, L)
    if z is not None:
        oys = oys * F.silu(z)
    return oys


def mamba_inner_ref(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    out_proj_weight,
    out_proj_bias,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
):
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_ref(
        x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, "silu"
    )
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, "b d l -> (b l) d"), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank : delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(
                B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2
            ).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(
                C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2
            ).contiguous()

   # old code
   # y = selective_scan_ref(
   #     x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True
   # )

   # ref code to compare to
    #y_ref = selective_scan_ref(
    #    x.detach().clone(), delta.detach().clone(), A.detach().clone(), B.detach().clone(), C.detach().clone(), D.detach().clone(), z=z.detach().clone(), delta_bias=delta_bias, delta_softplus=True
    #)

    # fast scan
    y = selective_scan_py_fast(
        x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True, chunksize=128, eps=1e-11
    )

    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)

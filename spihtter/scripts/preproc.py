"""
lauches the preprocessor over a web dataset shard list identified by braces
{000.100}.tar
"""
from typing import Any
import dask
import dask.distributed
from dataclasses import dataclass, asdict, field
from dask.distributed import get_worker
from dask.distributed import print as dprint
import braceexpand
import os
import torch
from diffusers import AutoencoderKL
import webdataset as wds
import tarfile
import json

import spiht

from spihtter.spiht_configuration import get_configuration
from spihtter.spiht_image import SpihtImage
from spihtter.utils import imsave

from ..dataset import get_wds_image_dataset, _SpihtImagePreprocessor, _SpihtVaePreprocessor


@dataclass
class PreprocessArgs:
    # Use the vae before compressing with spiht
    use_vae: bool = True
    # The old VAE works way better for some reason.
    vae_path: str = "stabilityai/sd-vae-ft-mse"  # or "madebyollin/sdxl-vae-fp16-fix"
    device: str = "cpu"
    dtype: str = "float32"  # or float16
    torch_dtype: Any = field(init=False)
    image_column_name: str = "jpg"
    bpp: float = 0.35
    # num dataset workers
    num_workers: int = 0
    max_res: int = 1024
    min_res: int = 128
    resume: bool = True

    spiht_configuration_mode : str = "BaseSpihtConfiguration"

    def __post_init__(self):
        self.torch_dtype = getattr(torch, self.dtype)
        self.spiht_configuration = get_configuration(self.spiht_configuration_mode)


class VaeProvider(dask.distributed.WorkerPlugin):
    def __init__(self, args: PreprocessArgs):
        self.args = args

    @staticmethod
    def _get_vae(args: PreprocessArgs):
        dprint("loading vae...")
        vae = (
            AutoencoderKL.from_pretrained(args.vae_path)
            .to(args.torch_dtype)
            .to(args.device)
        )
        return vae


    def setup(self, worker):
        if self.args.use_vae:
            vae = self._get_vae(self.args)
        else:
            vae = None
        worker._vae = vae


def _get_tarfile_basenames(shard):
    with tarfile.open(shard) as tf:
        names = tf.getnames()

    names = [x.split(".")[0] for x in names]

    return names


def _check_equal_rows(input_shard, output_shard):
    try:
        input_names = set(_get_tarfile_basenames(input_shard))
        output_names = set(_get_tarfile_basenames(output_shard))
    except:
        return False
    return input_names == output_names


def _proc_shard(input_shard, output_shard, args:PreprocessArgs):
    if args.resume:
        if os.path.exists(output_shard):
            if _check_equal_rows(input_shard, output_shard):
                dprint("Skipping ", output_shard)
                return
            else:
                dprint("Re-processing ", output_shard)

    ds = get_wds_image_dataset(
        input_shard,
        max_res=args.max_res,
        min_res=args.min_res,
        # probably want to change this for large encodings
        handler=wds.handlers.reraise_exception,
        image_column_name=args.image_column_name,
    )

    spiht_configuration = args.spiht_configuration

    vae = get_worker()._vae

    if args.use_vae:
        ds = ds.map(
                _SpihtVaePreprocessor(spiht_configuration, max_seq_len=None, bpp=args.bpp, vae=vae)
                )
    else:
        ds = ds.map(
                _SpihtImagePreprocessor(spiht_configuration,
                                        max_seq_len=None,
                                        bpp=args.bpp)
                )


    worker_id = get_worker().id

    with wds.TarWriter(output_shard, compress=True) as tarwriter:
        _tot = 0
        for row in ds:
            label = row.pop('label')
            if label is not None:
                row['cls'] = label

            encoding_result_dict = {}
            other_data = {}
            for k,v in row.items():
                if k.startswith("encoding_result_"):
                    encoding_result_dict[k] = v
                else:
                    other_data[k] = v

            encoding_result = spiht.EncodingResult.from_dict(encoding_result_dict)

            tarwriter.write({
                'encoding_result.pyd': encoding_result,
                **other_data
            })

            _tot += 1

            if _tot % 50 == 0:
                dprint(f"{worker_id} wrote {_tot} items in {output_shard}")


def _dask_preproc_shards(input_shards, output_shards, args: PreprocessArgs):
    client = dask.distributed.get_client()
    client.register_plugin(VaeProvider(args))

    print(f"processing {len(output_shards)} wds shards")

    delayed = []
    for input_shard, output_shard in zip(input_shards, output_shards):
        d = client.submit(_proc_shard, input_shard, output_shard, args, retries=10, pure=False)
        delayed.append(d)

    for d in delayed:
        d.result()

    print("done!")


def main(
    args: PreprocessArgs = PreprocessArgs(),
    shard_path: str = None,
    output_path: str = None,
    start_cluster: bool = False,
):
    if start_cluster:
        cluster = dask.distributed.LocalCluster(n_workers=1, threads_per_worker=1)
        client = cluster.get_client()

    shards = list(braceexpand.braceexpand(shard_path))
    shards_basenames = [os.path.basename(x) for x in shards]
    output_shards = [os.path.join(output_path, x) for x in shards_basenames]
    os.makedirs(output_path, exist_ok=True)

    args_d = asdict(args)
    del args_d["torch_dtype"]
    with open(os.path.join(output_path, "preprocess_args.json"), "w") as f:
        json.dump(args_d, f)

    return _dask_preproc_shards(shards, output_shards, args)


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)

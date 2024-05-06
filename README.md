# Spihtter

See [my article](https://theadamcolton.github.io/generative-modelling-of-compressed-image-file-bits)

Trains autoregressive models to predict the bits of an encoded image file

Currently only works well with mnist and llama. Mamba is a WIP

https://github.com/theAdamColton/spihtter/assets/72479734/e663c78e-a3fb-4607-a352-52840b104791

# Train a llama model that can generate mnist digits

You will need python>=3.11 as well as a rust compiler. `pip install -r requirements.txt` should do it. You can optionally install mamba from https://github.com/state-spaces/mamba to get access to the fast cuda implementation. This repo has a reference implementation of mamba in pure python/pytorch which is equivalent, but slower.

# download mnist as webdataset tar files

```bash
mkdir datasets/mnist/; cd mnist
huggingface-cli download clip-benchmark/wds_mnist --repo-type dataset --local-dir ./ --local-dir-use-symlinks False
```

# History

v0.0.1 - Mnist digit generation

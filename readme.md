# Re-distributed Sharded Data Parallel (RSDP)

## Saving Fine-tuning GPU Memory Usage by RSDP

RSDP reduces GPU memory usage by fine-grained pipelined computation and communication on the original fully sharded data parallelism (FSDP).

![Fully Sharded Data Parallelism](./img/fsdp.png)

![Re-distributed Sharded Data Parallelism](./img/rsdp.png)

In a word, RSDP removes the need of full materializtion of each single layer at computation, which is required by FSDP.

## Explainations

- [Original development repo](https://github.com/wdlctc/rtp) is following the first version of FSDP, which is based on the `FlattenParameter`.
- This repo aims to improve performance by using [PyTorch `DTensor`](https://github.com/pytorch/pytorch/tree/main/torch/distributed/tensor), following the exact way of [FSDP2](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md).

## Project Architecture

Following the architecture of FSDP2:
1. Design: [[RFC] Per-Parameter-Sharding FSDP #114299](https://github.com/pytorch/pytorch/issues/114299)
2. Implementation: https://github.com/pytorch/pytorch/tree/main/torch/distributed/fsdp/_fully_shard
3. An example way of calling APIs: [parallelize_llama.py in TorchTitan](https://github.com/pytorch/torchtitan/blob/7281e0be8feeb607f3c3f12cc3ceaafed87912c9/torchtitan/parallelisms/parallelize_llama.py#L336)

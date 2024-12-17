# Re-distributed Sharded Data Parallel (RSDP)

## Saving Fine-tuning GPU Memory Usage by RSDP

RSDP reduces GPU memory usage by fine-grained pipelined computation and communication on the original fully sharded data parallelism (FSDP).

![Fully Sharded Data Parallelism](./img/fsdp.png)

![Re-distributed Sharded Data Parallelism](./img/rsdp.png)

In a word, RSDP removes the need of full materializtion of each single layer at computation, which is required by FSDP.
In this way, RSDP mainly saves GPU memory usage from model weights, which can be particularly useful for fine-tuning large models on small datasets with few GPUs.

More specifically, RSDP aggressively re-distributes sharded model slices among GPUs and performs partial computation with the input for that GPU after each re-distribution. 
Finally, each shard would have visited every GPU at different time and we can get the full output.

### When to use RSDP

RSDP is useful when GPU memory is significantly occupied by model weights.
You can consider RSDP when
1. You have at least more than 1 GPU but not very many (total GPU memory size is not large).
2. Your expected training step number is not large (dataset is not large).

So RSDP is very suitable for fine-tuning, which often satisfies the abovementioned conditions.

### When NOT to use RSDP

1. You are not bounded by the aviable GPU memory size.
2. You want to accelerate training process by having a large batch size on each GPU.
    - In such case, the GPU memory is dominated by optimizer states instead of model weights.

## Project Architecture

Following the architecture of FSDP2:
1. Design: [[RFC] Per-Parameter-Sharding FSDP #114299](https://github.com/pytorch/pytorch/issues/114299)
2. Implementation: https://github.com/pytorch/pytorch/tree/main/torch/distributed/fsdp/_fully_shard
3. An example way of calling APIs: [parallelize_llama.py in TorchTitan](https://github.com/pytorch/torchtitan/blob/7281e0be8feeb607f3c3f12cc3ceaafed87912c9/torchtitan/parallelisms/parallelize_llama.py#L336)

## Explainations

- [Original development repo](https://github.com/wdlctc/rtp) is following the first version of FSDP, which is based on the `FlattenParameter`.
- This repo aims to improve performance by using [PyTorch `DTensor`](https://github.com/pytorch/pytorch/tree/main/torch/distributed/tensor), following the exact way of [FSDP2](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md).

Future work:
1. Since we have finer-grained computation and intermediate outputs, we can achieve finer-grained checkpoint activations.
2. Also finer-grained pipeline for CPU offloading. 

## Benchmarks

### RTP

```bash
python benchmarks/rtp/dp_benchmark.py --use_synthetic_data
python benchmarks/rtp/fsdp_benchmark.py --use_synthetic_data
python benchmarks/rtp/tp_benchmark.py --use_synthetic_data
python benchmarks/rtp/rtp_benchmark.py --use_synthetic_data
```

Config:

```bash
python benchmarks/rtp_benchmark.py \
    --max_batch 10 \
    --use_synthetic_data \
    --vocab_size 10240 \
    --ninp 2048 \
    --nhid 2048 \
    --nhead 32 \
    --num_decoder_layers 10 \
    --seq_len 32 \
    --batch_size 32 
```

### RSDP

WIP.
# Batch-shape experiments: runbook

One command per experiment and arm; every command takes a results
directory. Install the hook once per machine and per virtualenv:

```
pip install -e probes/shape/shape_hook_pkg      # registers the vllm.general_plugins entry point
```

The hook is inert unless `SHAPE_HOOK_OUT` is set, which the runners do.
Engine sha and wheel: the runners write the environment record the other
probes write (GPU, driver, CUDA, torch, vLLM and FlashInfer versions, env
flags). The hook targets vLLM 0.28.0; line references in its docstring are
at the census sha 5769a7382cb1.

## E2, bucket attribution (multiprocess engine, stock scheduler)

```
for g in 1 0; do for p in 1 0; do
  python probes/shape/run_e2.py --model Qwen/Qwen2.5-7B-Instruct --cudagraph $g --prefix-caching $p --repeats 12 --out probes/shape/results/e2
done; done
python probes/shape/run_e2.py --model Qwen/Qwen1.5-MoE-A2.7B-Chat --cudagraph 1 --prefix-caching 1 --repeats 12 --out probes/shape/results/e2
```

About 4 minutes per arm on one H100 (model load plus 12 generations of
16 requests). The MoE arm is the X1 check; its summary lists the
per-expert counts alongside the shape histories.

## E3, logged replay (in-process engine, staged arrival)

```
RUN_TAG=a python probes/shape/run_e3.py --model Qwen/Qwen2.5-7B-Instruct --out probes/shape/results/e3
RUN_TAG=b python probes/shape/run_e3.py --model Qwen/Qwen2.5-7B-Instruct --out probes/shape/results/e3
python probes/shape/run_e3.py --compare probes/shape/results/e3/Qwen_Qwen2.5-7B-Instruct_tp1_none_graphs1_prefix1
```

About 3 minutes per tag. Step 0 is prefill, step 5 is a mixed batch
(eight decodes plus eight prefills), later steps are decode. The
comparison reports those and every step.

## E4, dummy-neighbour replay (in-process engine, prefix caching off)

```
R=probes/shape/results/e4
python probes/shape/run_e4.py --model Qwen/Qwen2.5-7B-Instruct --repeats 12 --out $R                       # bf16, graphs on
python probes/shape/run_e4.py --model Qwen/Qwen2.5-7B-Instruct --repeats 12 --cudagraph 0 --out $R         # bf16, graphs off
python probes/shape/run_e4.py --model Qwen/Qwen2.5-7B-Instruct --repeats 12 --tp 2 --out $R                # bf16, TP=2 (two GPUs)
python probes/shape/run_e4.py --model Qwen/Qwen2.5-7B-Instruct --quantization fp8 --repeats 12 --out $R    # FP8 per-token dynamic (vLLM default with CUTLASS)
python probes/shape/run_e4.py --model Qwen/Qwen2.5-7B-Instruct --quantization fp8 --fp8-per-tensor --repeats 12 --out $R   # FP8 per-tensor dynamic (predicted failure)
python probes/shape/run_e4.py --model Qwen/Qwen3-8B-FP8 --repeats 12 --out $R                              # FP8 blockwise
python probes/shape/run_e4.py --model Qwen/Qwen1.5-MoE-A2.7B-Chat --repeats 12 --out $R                     # MoE (predicted failure, routing)
```

About 6 minutes per arm on one H100 (24 generations). TP=2 needs two
GPUs and NCCL_NVLS_ENABLE=0 in a RunPod container.

## Mechanism arms (eager model)

`torch.compile` inlines the model, so module hooks and the Python wrappers
of custom ops do not run; the per-expert routing counts and the dynamic
FP8 scale are therefore only recorded by the eager arms:

```
python probes/shape/run_e4.py --model Qwen/Qwen1.5-MoE-A2.7B-Chat --repeats 6 --no-compile --cudagraph 0 --out $R
python probes/shape/run_e4.py --model Qwen/Qwen2.5-7B-Instruct --quantization fp8 --fp8-per-tensor --repeats 6 --no-compile --cudagraph 0 --out $R
```

## Re-analysing saved arms

The runners' joins can be recomputed offline from a saved arm directory:

```
python probes/shape/run_e2.py --analyse probes/shape/results/e2/<arm>
python probes/shape/run_e4.py --analyse probes/shape/results/e4/<arm>
python probes/shape/run_e3.py --compare probes/shape/results/e3/<arm>
```

## E5, cross-SKU

Run the E3 tag `a` command on each GPU into its own results root, then:

```
python probes/shape/run_e5.py results_h100/e3/<arm>/a results_b200/e3/<arm>/a
```

## Tables

```
make shape-tables        # python probes/shape/tables.py probes/shape/results
```

# Retained graph observations

These observations motivated the boundary recorder. They are source facts and
candidate discriminators, not a cause finding.

The successful CUDA 12.9 and 13.0 V2 compiled archives each retain one vLLM
`backbone/computation_graph.py`. The files are both 528,461 bytes. Their raw
SHA-256 values differ because source comments contain the environment path
`/opt/issue56900/cu129/` or `/opt/issue56900/cu130/`. Replacing only that path
component makes the files byte-identical, with normalized SHA-256:

```text
5f141a7041361ebee636c40a3a5d0a80f59e042a4c399c20e2a845de79f2c845
```

Each graph contains 24 assignments that invoke the opaque custom op:

```python
torch.ops.vllm.moe_forward_shared(view_1, linear_1, view_1, None, layer_name, 0)
```

The first and third tensor arguments are the same `view_1`: routed and shared
experts receive the same hidden-state view. The two returned tensors are added
in the compiled graph. The retained graph source therefore supports recording
these boundaries:

1. the hidden-state/shared input and router logits entering each opaque op;
2. top-k IDs and weights computed inside the op;
3. the shared and routed tensors returned to the compiled graph; and
4. whether the input changes while the custom op runs.

In vLLM tag `v0.28.0` at commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a`, `moe_forward` is registered with
`mutates_args=["hidden_states"]`, while `moe_forward_shared` is registered
without a mutation declaration. That difference alone does not prove a schema
bug: the shared path may correctly leave its inputs unchanged. The localiser
captures exact pre/post bytes so the GPU observation can decide that narrower
question.

The graph also computes post-attention normalisation and the 60-way router
linear outside the opaque MoE op. Consequently:

- different `moe_input` at layer 0 places the earliest observed difference
  upstream of the first MoE call;
- equal `moe_input` but different `router_logits` isolates the gate linear;
- equal inputs/logits but different top-k or returned outputs places the first
  observed difference inside the opaque MoE path; and
- equal returned shared/routed outputs followed by different next-layer input
  brackets the difference to the compiled add/residual/attention region between
  those two boundaries.

The recorder reports a boundary, not an individual generated Triton kernel.
Raw-tensor replay is deferred until the first differing boundary is known.

## Generated storage reuse discriminator

The retained CUDA 13.0 Inductor source
`cache/torchinductor/wi/cwibatuzlvf3ethgjt4ho5zm34bmyyxxdsdsyqoky3wwtj26ilyu.py`
contains a narrower scheduling observation. At source lines 931–948 it creates
`buf2` for the normalized hidden state, uses it for the 60-way gate matrix
multiplication, and passes it as both tensor inputs to `moe_forward_shared`.
After extracting both custom-op results, line 960 assigns
`buf8 = buf2; del buf2  # reuse`; lines 962–964 then write the combined
shared/routed add and next RMS normalisation into that storage.

This does not establish an invalid reuse. It is valid if all custom-op reads
and aliases have finished with the storage, including work on any internal
stream. It does give two cheap follow-ups if the recorder changes the compiled
tokens: a post-op synchronization-only control, and compilation with both
`torch._inductor.config.inplace_buffers` and `allow_buffer_reuse` disabled.
The generated source must be inspected to confirm that the latter control
actually removes this reuse.

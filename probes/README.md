# Runtime probes

Scripts that settle what static analysis cannot. None of them was run in
producing this census; they are written against the pinned shas and the
public APIs of each engine, and they need one or more NVIDIA GPUs. Each
prints a verdict line of the form `PROBE <name> <bitwise-identical|DIFFERS>`
and writes the raw tensors it compared next to the script.

Common protocol (`probes/common.py`):

1. fix seeds, `CUBLAS_WORKSPACE_CONFIG`, and the engine configuration;
2. run the computation twice in the same process and once more in a fresh
   process;
3. compare outputs with `torch.equal` on the raw bytes (`view(torch.int32)`
   for floats), never with a tolerance;
4. record GPU SKU, driver, library versions and the pinned sha in the
   output JSON so that a mismatch can be attributed.

See `docs/RUNTIME_PROBES.md` for which inventory row each probe settles.

# Native OpenPI Pi0.5 LIBERO contract — experimental

The native domain uses H=10, `discrete_state_input=False`, and no extra delta
transform. Its normalization, image operations and tokenizer must be compared
with the official policy together. H=50 → H=10 is a workload change, never a
same-workload engine speedup.

Sources checked on 2026-11-14:

- [Official configuration](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)
- [SentencePiece prompt behavior](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py)
- [LIBERO cameras and output dimensions](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/libero_policy.py)
- [Normalization and transforms](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py)
- [PIL image resize](https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/image_tools.py)

`vvla_pi05.openpi` implements a separate processor. It takes the original local
SentencePiece model, preserves BOS and the separately tokenized newline, orders
base/left-wrist/masked-right-wrist cameras, resizes uint8 RGB using PIL before
SigLIP normalization, uses OpenPI quantile epsilon 1e-6, and returns the first
seven unnormalized action dimensions without adding robot state. It does not
route OpenPI observations through the LeRobot processor.

The release loader still requires an explicit converted Torch checkpoint and
resolved config. Strict weight loading remains enabled. No native OpenPI full
checkpoint has been qualified in this local round. CPU contract tests do not
replace token/image/prepared-input parity against the installed official policy,
explicit-noise full-model parity, or paired closed-loop evaluation. Benchmark
configs must identify this domain independently and record the upstream commit,
checkpoint/config/tokenizer/norm-stats hashes before remote execution.

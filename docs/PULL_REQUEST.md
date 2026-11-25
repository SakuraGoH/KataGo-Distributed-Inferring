# Distributed inference: remote NN servers over TCP

## Summary

This branch adds a distributed inference mode to KataGo: the engine keeps all search state
and forwards every neural-net batch over TCP to one or more remote `katago distserver`
processes, each of which owns one or more GPUs and runs a normal local backend
(CUDA/TensorRT/OpenCL/Eigen/ONNX/ROCm). The design follows PhoenixGo's
`dist_zero_model_server`: servers are stateless evaluators, the engine stays single-host.

```
engine (CPU-only host) --TCP--> distserver A (GPU host 1)
                      \-------> distserver B (GPU host 2)
```

What is included:

* `katago distserver` subcommand — serves evaluations, one connection per engine NN
  server thread, one compute handle per connection, `-gpus 0,1,...` round-robins
  connections across devices, `-max-batch` and `-boardsize` bound what it serves.
* `-DUSE_BACKEND=DISTRIBUTED` client backend — no local NN, only the model header is
  parsed; `distServers = host:port,...` plus `numNNServerThreadsPerModel` control how many
  connections are opened and how they rotate over the servers.
* Wire protocol `KGD1` v2: model sha256 + board size + batch cap are checked at the
  handshake; the server reports its effective post-processing scale (see below).
* Docs (`docs/DistributedInference.md`), a 26-check local harness
  (`dist_test/feasibility/`), and the two-machine cloud scripts + report
  (`dist_test/feasibility/cloud/`).

## Validation (2 rented GPU nodes, public internet)

Two auto-scaling cloud instances, one RTX 4080 SUPER 32 GB each, CUDA 13.0 / cuDNN 9:

* both machines built from scratch (2 m 28 s each, `-j48`);
* one `distserver` per machine, bound to `127.0.0.1:17731`;
* one SakuraLink SSH tunnel per machine, engine configured with
  `distServers = 127.0.0.1:<a>,127.0.0.1:<b>`;
* both servers logged real work (48 / 37 `eval batch` lines), and the remote evaluation is
  **bit-identical** to same-machine direct inference (`whiteWin 0.440201`, `whiteLead 0.550`);
* throughput at batch 8: GPU-direct 3127 evals/s, one node over the tunnel 56 evals/s,
  two nodes 62 evals/s (2 connections) and 113 evals/s (4 connections).

Video (2 min, from a clean build to the benchmark comparison):
[docs/media/kata-distributed-verify.mp4](https://github.com/SakuraGoH/KataGo-Distributed-Inferring/blob/docs/distributed-inference/docs/media/kata-distributed-verify.mp4)

## Bug fixed during validation

The first cross-machine runs produced wrong values for convnet models (empty board
`whiteWin` 0.4925 instead of 0.4402), while transformer models matched exactly. Root cause:
CUDA/ROCm/OpenCL/TensorRT/ONNX backends replace the model with an activation-scaled variant
inside `createComputeContext` (`applyScale8ToReduceActivations`) and multiply
`postProcessParams.outputScaleMultiplier` by 8. The server sent logits in that 8x scale, but
the client only parses the model header and post-processed them with the unscaled value (1).
Transformer models are never scale8'd, which is why they matched.

Fix (protocol v2): the server reports its effective `outputScaleMultiplier` in the handshake
and the client rescales policy/value/score tensors by `serverScale / clientScale` before its
own post-processing. After the fix, convnet b12c192, convnet b10c128, a transformer model and
NCHW clients all match same-machine direct inference exactly.

## Limitations

* **Network-bound.** One round trip per batch, plus 32.6 KB per eval upstream. On a 24 ms
  RTT link with ~10 MB/s upload a two-machine fleet tops out at ~100–350 evals/s while the
  GPUs can do thousands; the GPU sits below 5% utilisation. Same host or 10 GbE is required
  to use the GPUs fully, and a 1 GbE LAN still caps this small model at ~3.8k evals/s.
* **Protocol v2 is not backward compatible** — client and server must be rebuilt together.
* The server has **no config file**, so backend options (`cudaShareModelWeights`,
  `openclTunerFile`, FP16 overrides, …) cannot be set on the server side.
* Every connection owns its own model copy on the GPU; too many connections per GPU exhaust
  VRAM.
* **No reconnect or failover**: if a server dies, the engine aborts; the fleet must be
  restarted as a unit.
* **No authentication or TLS** on the wire; the intended deployment keeps the server bound
  to loopback and reaches it through SSH.
* The bundled paramiko tunnel is convenient but caps at ~2.4 MB/s; system `ssh -L` reaches
  ~10–13 MB/s and is what a high-throughput deployment should use.

## Why there is no suitable experimental setup yet

This branch is a working prototype, not something ready to merge upstream:

* The only available hardware was **single-GPU rental instances**; the multi-GPU-per-machine
  path (`-gpus 0,1`) is implemented but was never exercised end-to-end.
* All measurements were **across the public internet** (24–32 ms RTT). No same-LAN or
  same-host multi-GPU pair was available, so the "network is the limit" conclusion could not
  be separated from the cluster's own performance. A proper evaluation needs a 2–4 GPU
  machine with the engine on the same host, then a same-LAN pair, then a WAN pair.
* The code lives on a fork with API changes (batch policy hooks, extra backend options) and
  **no shared git history with upstream**, so the feature cannot be cherry-picked as-is; it
  needs a rebase onto `master` and a review of the backend API it relies on.
* The benchmark model available for testing was a small custom net (b12c192); a realistic
  b18/b28 measurement on modern GPUs was not possible within the rented time.

## Follow-ups

1. Rebase the feature onto upstream `master` and drop the fork-specific API dependencies.
2. Add a `-config` option to `distserver` for backend tuning (FP16, workspace, tuner files).
3. Reuse the engine's warmup path for transformer models on the server side.
4. Add reconnect/retry and consider a small compression/quantisation option for the wire.
5. Validate on a 2-GPU host (engine on the same host) and on a 10 GbE LAN pair.

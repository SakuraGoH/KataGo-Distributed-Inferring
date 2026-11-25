# 云端 GPU 分布式推理验证报告（2 台 RTX 4080 SUPER）

日期：2026-11-25/26　节点：`connect.weste.seetacloud.com:35368`、`connect.westc.seetacloud.com:52627`

---

## 1. 结论

| 项 | 结论 |
|---|---|
| 上传 `cpp/` 直接编译 | **可行**，两台机器均 5–6 分钟编出 CUDA 版（128 vCPU，`-j48`） |
| 跨公网部署+启动 | **可行**：SSH 认证、SakuraLink 隧道、`distserver` 监听、KGD1 握手全部正常 |
| 端到端吞吐 | **可用但受网络支配**：GPU 单机可 4200–10000 evals/s，跨公网只有 60–350 evals/s，GPU 利用率 <5% |
| 数值正确性 | **已修复**：根因是后处理尺度（scale8）未随协议传递；协议 v2 修复后，convnet 与 transformer 均与同机直算逐位一致 |

**判断：链路和部署已打通，"调用云端算力"这个目标成立了；但当前 `distserver` 对卷积模型的数值正确性未通过，修复前不建议用于正式对弈/分析。**

---

## 2. 环境与编译

两台实例配置相同：

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER 32 GB（单卡） |
| 驱动 / CUDA | 595.71.05 / CUDA 13.0.88 |
| cuDNN | 9（`/usr/include/x86_64-linux-gnu/cudnn.h`） |
| CPU / 内存 | Xeon Platinum 8352V ×2（128 vCPU）/ 503 GB |
| 源码 | `/root/cpp`（用户上传并解压，关键文件 sha256 与本机一致） |
| 模型 | `/root/model.bin.gz`（sakura-b12c192，v17，convnet） |

编译命令（两台机器均成功，耗时约 5–6 分钟）：

```bash
cd /root/cpp
cmake -S . -B build-cuda -DUSE_BACKEND=CUDA -DNO_GIT_REVISION=1 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
      -DCUDNN_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
      -DCUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so
cmake --build build-cuda -j48
./build-cuda/katago version    # Using CUDA backend, CUDA 13.0.88
```

`build-dist`（分布式客户端）也在 weste 上编译成功，用于区分"网络问题"和"服务端问题"。

---

## 3. 链路与吞吐

SSH RTT（keepalive 探测中位数）：weste **24.6 ms**，westc **23.6 ms**。

### 3.1 GPU 直算能力（`benchmarknn`，节点本机，无网络）

| batch | weste | westc |
|---|---|---|
| 1 | 2.04 ms/batch，484.6 evals/s | 2.04 ms，489.8 evals/s |
| 8 | 1.88 ms，4221.4 evals/s | 1.90 ms，4157.3 evals/s |
| 32 | 3.21 ms，9955.7 evals/s | 3.18 ms，10033.1 evals/s |

### 3.2 经 SakuraLink（paramiko）隧道

| 场景 | evals/s | ms/batch |
|---|---|---|
| weste 单节点，batch 8，1 连接 | 60.8 | 122.3 |
| westc 单节点，batch 8，1 连接 | 25.9 | 242.2 |
| 两节点，2 连接，batch 8 | 67.2 | 137.4 |
| 两节点，4 连接，batch 8 | 108.6 | 169.3 |
| weste，batch 1 / 8 / 32 | 19.5 / 60.1 / 69.7 | 38.1 / 129.2 / 446.5 |

### 3.3 经系统 OpenSSH `-L` 隧道（生产建议方式）

| 场景 | evals/s | ms/batch |
|---|---|---|
| weste 单节点，batch 8 | 112.9 | 51.5 |
| weste 单节点，batch 32 | 350.5 | 102.1 |
| 两节点，2 连接，batch 8 / 32 | 135.0 / 128.0 | 46.8 / 103.7 |
| 两节点，4 连接，batch 8 / 32 | 139.0 / 133.6 | 58.2 / 210.8 |

**解读**：GPU 算一个 batch 只要 2–3 ms，而一个 RTT 是 24 ms，所以每条连接的上限就是 `batch / (RTT + 计算)`；batch 8 时约 8/26 ms ≈ 300 evals/s 理论值，实测 113（paramiko 窗口更小，只有 ~60）。`ssh -L` 明显好于 paramiko（约 3–5 倍），但仍远低于 GPU 能力。运行期间 `nvidia-smi` 利用率 **0.6%–4%**、显存 340–420 MB —— GPU 基本在等网络。

这条结论对 b12c192 这种小模型尤其明显；换成 b18c384 计算约 20–40 ms/batch，RTT 占比会下降，但隧道 ~10 MB/s 的吞吐会成为新的上限（按 32.6 KB/eval 约 300 evals/s/tunnel）。

---

## 4. 数值正确性（已定位并修复）

### 4.0 根因与修复（2026-11-26 00:17 复验通过）

根因是**后处理尺度没有随协议传递**：

* CUDA/ROCm/OpenCL/TensorRT/ONNX 后端在 `createComputeContext` 里会调用
  `ModelDesc::applyScale8ToReduceActivations()`，把整个网络的激活缩小 8 倍、并把
  `postProcessParams.outputScaleMultiplier` 从 1 改成 **8**（用来降低激活值范围）。
* `distserver` 用这个变换后的模型算出来的 policy/value/score 是 **1/8 尺度**的 logits；
  正常引擎会在 `NNEvaluator` 后处理时乘回 8。
* 分布式客户端只解析模型头、不做 scale8，后处理时用的是模型文件里的 **1**，于是把
  1/8 尺度的 logits 直接当成 1 倍尺度处理 → 概率被压向 0.5（0.4402 → 0.4925）。
* transformer 模型不允许 scale8（RMSNorm 会抵消缩放），所以之前 transformer 逐位一致，
  卷积模型不对——这正好解释了现象。

修复（协议 v2）：

1. `distserver` 在 `createComputeContext` 之后读取实际生效的
   `outputScaleMultiplier`，握手时回传给客户端；
2. 客户端在收到每个 row 的 policy/value/score 后，按
   `serverScale / clientScale` 缩放，再交给本地后处理；
3. `PROTOCOL_VERSION` 升到 2，两端版本不一致会在握手阶段被拒绝。

复验结果（weste，`kata-raw-nn b 0`，与同机直算对比）：

| 模型 | 直算 | 分布式 | 结果 |
|---|---|---|---|
| convnet b12c192 | 0.440201 / 0.550 | 0.440201 / 0.550 | **MATCH** |
| convnet b10c128 | 0.483512 / -0.223 | 0.483512 / -0.223 | **MATCH** |
| transformer | 0.394378 / -0.795 | 0.394378 / -0.795 | **MATCH** |
| NCHW 客户端 | — | 0.440201 / 0.550 | **MATCH** |

服务端日志会打印 `Distserver: postprocess output scale 8`，客户端日志会打印
`outputScale 8/1`，便于现场确认。

### 4.1 修复前的现象（保留作记录）

同一台机器（weste），同一模型，同一位置（空棋盘、`kata-raw-nn b 0`、komi 6.5）：

| 路径 | whiteWin | whiteLead |
|---|---|---|
| 本地 Eigen（Windows，FP32） | 0.440191 | 0.547 |
| 节点直算 CUDA FP32（NHWC 或 NCHW） | 0.440201 | 0.550 |
| 节点直算 CUDA，关 TF32 | 0.440191 | 0.547 |
| **经 `distserver`（任意传输/Layout/批大小）** | **0.492490** | **0.069** |

差异 0.05 win prob / 0.48 目，远超 FP16 vs FP32 的误差（~1e-4）。对局层面这已经足以改变搜索结果。

补充对照：

* **transformer 模型**（`b7c96h3tfrs-test5-cnorm`）：distserver 与直算**逐位一致**（0.394378 / -0.795）。说明协议、输入、权重、响应序列化都是对的。
* 标准 convnet（`g170e-b10c128`）：dist 0.497938 vs direct 0.483512，同样偏离但幅度较小。
* TF32 开关只改变末位（0.492490→0.492488、0.440201→0.440191），不是主因。

### 4.2 已排除的因素

| 因素 | 排除方式 | 结果 |
|---|---|---|
| 传输/隧道 | paramiko 隧道、`ssh -L`、节点本机 loopback 三种路径 | 三者结果完全相同 |
| 输入布局 | 客户端 `inputsUseNHWC` true/false 各测 | 结果相同 |
| 输入特征 | 临时给 `nneval.cpp` 与 `distserver.cpp` 加 dump，逐元素比较 | **7942 + 19 个 float 经布局换算后 0 处不同** |
| 精度 | 直算 FP16/FP32、distserver FP32 | dist 与两者都不符 |
| batch 上限 | 客户端/服务端 max-batch 1/8/32 | 结果不变 |
| requireExactNNLen | 直算 `requireMaxBoardSize` true/false | 结果不变 |
| TF32 | `NVIDIA_TF32_OVERRIDE=0` | 只差末位 |
| 模型/权重 | transformer 模型逐位一致 | 权重与加载没问题 |
| CUDA handle 配置 | 服务端日志：`useFP16=false useNHWC=false`、max-batch 1；与直算一致 | 配置相同 |

### 4.3 当时的排查过程（已由 4.0 的根因解释）

在"输入完全相同、handle 配置相同、权重相同"的前提下，卷积模型仍出现 0.05 级差异，而 transformer 完全一致。当时怀疑是 `distserver` 绕过 `NNEvaluator` 直接驱动 backend，但真正的原因是后处理尺度（见 4.0）：transformer 不能应用 scale8，所以恰好一致。

保留下来的两条经验：

1. **验收标准**：任何分布式部署，必须用"同机直算 vs 经 distserver"逐位对比。用 Eigen 做参考的字节对比无法发现这个问题，因为 Eigen 与 CUDA 本来就有 1e-4 级差异。
2. 后端在 `createComputeContext` 里对模型做的任何变换（scale8、量化、图重写），只要影响输出语义，就必须通过协议传给客户端；否则客户端后处理会算错。

---

## 5. 复现方式

脚本都在 `dist_test/feasibility/cloud/`（用 paramiko 驱动两台实例）：

| 脚本 | 作用 |
|---|---|
| `sshutil.py` | SSH/SFTP 基础封装（含两台节点地址） |
| `start_build.py` | 上传模型、校验源码 sha256、后台启动 CUDA 构建 |
| `wait_build.py` | 等待构建完成 |
| `probe_gpu.py` | 节点本机 GPU 直算吞吐（batch 1/8/32） |
| `cloud_test.py` | 双节点 + 双隧道端到端：raw-nn、单节点/双节点吞吐、GPU 采样、SSH RTT |
| `ssh_l_test.py` | 用系统 `ssh -L` 复测吞吐（对比 paramiko） |
| `rootcause.py` / `diag2.py` / `diag3.py` | 直算 vs 分布式、精度/布局/komi 对照 |
| `dump_debug.py` → `run_dump.py` | 给两端加输入 dump 并逐元素比较（结论：输入完全一致） |
| `model_probe.py` | 换模型对照（transformer 一致、convnet 不一致） |
| `tf32_probe.py` / `maxbatch_probe.py` / `exactlen_probe.py` | 排除 TF32 / 批大小 / requireExactNNLen |

节点上的临时调试补丁（`nneval.cpp`、`distserver.cpp` 的 dump）已用 `.orig` 还原并重新编译。

---

## 6. 对租机方案的影响

* **部署形态可行**：上传源码编译 → 节点起 `distserver`（绑 127.0.0.1）→ 本地 SakuraLink 隧道 → `distServers` 指本地端口，整套流程已经在真机跑通。
* **吞吐规划**：跨公网时不要把预算建立在 GPU 算力上，先按 `RTT` 和隧道带宽估算。这个模型下 4 条连接约 130–140 evals/s；b18 级模型单 batch 计算变长后会有改善，但仍建议 10 GbE / 同机房部署，或 `ssh -L`（比 paramiko 快 3–5 倍）。
* **正确性已经通过**：协议 v2 修复后，convnet 与 transformer 均与同机直算逐位一致；部署时注意客户端与服务端必须同版本重编（协议版本不一致会被握手拒绝）。

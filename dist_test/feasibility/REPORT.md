# KataGo 分布式推理引擎 —— 多机多卡可行性验证报告

日期：2026-11-25　目标机：单台 Windows 笔记本（无 NVIDIA GPU）
被测对象：`cpp/neuralnet/distributedbackend.cpp`（客户端后端）+ `cpp/command/distserver.cpp`（推理服务端）+ `cpp/neuralnet/distprotocol.{h,cpp}`（KGD1 协议 v1）
验证脚本：`dist_test/feasibility/harness.py`（可重复执行）
原始结果：`dist_test/feasibility/out/results.json`、`out/logs/*.log`

---

## 1. 结论

| 维度 | 结论 | 证据强度 |
|---|---|---|
| 功能正确性 | **通过**。远端推理与本地推理的 raw-NN 输出逐字节一致（含 8 种对称变换），固定随机种子的 6 手对局逐步一致 | 强（端到端实测） |
| 多机 | **通过**。3 个独立服务进程（模拟 3 台机器）经真实 TCP 工作；非回环网卡地址可用；两台机器可同时服务两个引擎；故障语义明确 | 强（端到端实测） |
| 多卡 | **机制通过，真卡未验**。`-gpus 0,1,2` 按连接轮转、每连接独立 compute handle 并在线程内绑定设备；本机无 NVIDIA GPU，CUDA 实际多卡未运行 | 中（调度+代码路径；缺真卡实测） |
| 性能 | **可用**。单 batch 传输开销实测 1–11 ms（回环），每 eval 上行 31.9 KB；1 GbE 上限约 3.7k evals/s；RTT 按 batch 计费 | 强（同一次运行的对照测量） |
| 运维/安全 | **有明确缺口**：无重连/故障转移、无 TLS/鉴权、服务端无配置文件 | 强（代码+故障注入） |

**总判断：多机方向可行且数据通路已经验证；多卡方向在代码与调度层面成立，但必须在真实 NVIDIA 机器上补一次 2 卡/2 机实测才能下最终结论。** 当前实现适合"引擎与 GPU 机器同局域网、由同一套进程管理统一拉起、接受服务器故障即整体失败"的场景；不适合把 GPU 通过公网/不可信网络暴露，也不适合要求单点故障不中断的在线服务。

本次验证还发现并修复了 5 个缺陷（其中 2 个会导致静默错误结果或工具不可用），见第 5 节。

---

## 2. 验证环境与方法

**环境**

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 5 125H（4 P-core + 8 E-core + 2 LP E-core，18 线程） |
| 内存 | 31.5 GB |
| 网络 | Intel Wi-Fi 6 AX201，LAN 地址 172.20.67.52 |
| GPU | 无 NVIDIA GPU（仅 Intel Arc 核显，未构建 OpenCL 后端） |
| 模型 | `model.bin.gz`，b12c192，V7 输入特征（22 平面），model version 15 |
| 服务端二进制 | `build-eigen/Release/katago.exe`（Eigen CPU 后端，模拟"GPU 机器"） |
| 客户端二进制 | `build-dist/Release/katago.exe`（`-DUSE_BACKEND=DISTRIBUTED`） |

**方法**

1. **用多个 `katago distserver` 进程模拟多台机器**：每台"机器"是一个独立进程 + 独立端口，客户端只通过 `distServers = host:port,...` 与它们交互。对客户端而言，3 个进程与 3 台局域网机器没有协议差异。
2. **多卡**用 `-gpus 0,1,2` 触发服务端的连接→设备轮转调度（Eigen 忽略设备号，但分配逻辑与 CUDA 完全相同），并用"每 GPU 一个进程"的部署方式再验证一次。
3. **传输开销用同一次运行内的对照**：客户端测一个 batch 的往返时间，服务端日志同时记录该 batch 的纯计算时间，两者相减即为传输+序列化开销。这样可排除两个进程落在不同核心/不同频率上造成的偏差（本机 P/E 核混排，跨进程对比会产生 2–3 倍噪声）。
4. **网络条件**用用户态 TCP 代理注入时延（每个 chunk 单向延迟 d，RTT≈2d）并统计字节数。
5. **正确性判据**：`kata-raw-nn COLOR all` 输出逐字节比较（覆盖 8 种对称、policy optimism、ownership）；以及固定 `nnRandomize=false` + `searchRandSeed` 的 6 手 genmove 对局比较。

---

## 3. 正确性结果

| 检查 | 结果 |
|---|---|
| 本地 Eigen 参考输出 | 133,632 字节 raw-NN 输出 |
| 经 1 台远端服务器 | **逐字节一致** |
| 经 3 台服务器（3 条连接） | **逐字节一致**，负载分摊到多台（batches/server = [1,7,10]） |
| NCHW 客户端 → 仅支持 NHWC 的服务端 | **逐字节一致**（修复后；修复前静默错误，见 5.1） |
| 6 手确定性对局（本地 vs 远端） | 手顺完全一致：`R16 D3 C16 R4 P3 Q3` |
| 两个引擎共享同一个 2 机集群 | 两者输出均逐字节一致，两台机器都实际参与计算 |

**为什么必须固定 `searchRandSeed`**：KataGo 的对局在 `numSearchThreads=1` 下也并非跨进程可复现——搜索随机数默认用时间种子（`gtp.cpp`：`searchRandSeed = uint64(seedRand.nextUInt64())`），节点哈希/树结构会随之变化。只设 `nnRandomize=false` 时，同一份本地配置连跑 3 次得到 3 种不同手顺（实测）。现有 `dist_test/run_tests.sh` 的对局 diff 因此并不可靠（其保存的输出 `out/game_eigen.txt` 与 `out/game_dist1.txt` 本身就是不同的手顺）。正确做法是同时固定 `nnRandomize=false` 与 `searchRandSeed`。

---

## 4. 多机 / 多卡 / 故障结果

### 4.1 多机

| 检查 | 结果 |
|---|---|
| 服务端绑定 LAN 地址 172.20.67.52，客户端经该地址连接 | 通过（非回环地址、真实网卡栈） |
| 工作分摊到集群 | 通过（批次由共享队列动态领取，慢机自然少领） |
| 两台机器同时服务两个引擎 | 通过，两边都实际计算（batches 25/11） |
| 第二个引擎在第一个搜索中途被服务 | 通过：短客户端 2.17 s 完成，长客户端仍在运行 |

### 4.2 多卡（调度机制）

| 检查 | 结果 |
|---|---|
| `-gpus 0,1,2` + 6 条连接 | connId→gpu = (0,0)(1,1)(2,2)(3,0)(4,1)(5,2)，严格轮转 |
| 三块设备都有实际求值 | 通过 |
| `-gpus 0` + 4 条连接 | 全部绑定 gpu 0 |
| 每 GPU 一个进程（`-gpus 0` / `-gpus 1`） | 两个进程都服务了请求 |

**CUDA 侧绑定路径（代码核对）**：`createComputeContext()` 对 CUDA 不使用 `gpuIdxs`（无需预初始化设备）；真正的绑定发生在 `createComputeHandle(..., gpuIdxForThisThread, ...)` → `CUDA_ERR("createComputeHandle", cudaSetDevice(gpuIdxForThisThread))`（`cudaandrocmbackend.inc:5096-5102`）。`distserver` 为每条连接新建线程并在该线程内创建 handle，因此"连接→线程→设备"三者一致，这与 KataGo 正常的"每 GPU 一个 NN server 线程"模型相同。

**未验证/需注意**：每条连接持有独立的 compute handle，即**每连接一份模型权重**（显存按连接数翻倍，`cudaShareModelWeights` 在服务端无法配置，见 6.3）；服务端未调用 `maybeWarmupComputeHandle`，CUDA transformer 模型（b18/b28 nbt）在新 batch size 上会触发 cuDNN 计划编译（见 6.4）。

### 4.3 故障与握手

| 检查 | 结果 |
|---|---|
| 启动时服务器不可达 | 2.8 s 内报 `could not connect to host:port` 并退出（失败快、信息明确） |
| 握手：magic 错误 | 拒绝（返回 status=1；若客户端还有未读数据，Windows 会 RST 掉错误文本，但连接确实被拒） |
| 握手：棋盘尺寸不匹配 | 拒绝，错误信息含服务端尺寸（`server serves 9x9`） |
| 握手：模型 sha256 不匹配 | 代码路径存在同样校验（未构造第二份模型，未实测） |
| 客户端 `nnMaxBatchSize` > 服务端 `-max-batch` | 修复后在握手阶段即拒绝并给出可操作提示（修复前会在对局中途断链，见 5.3） |
| 客户端断开 | 服务端记录日志并继续服务下一个客户端 |
| 搜索中途杀掉服务端 | 客户端立即以 `ERROR: NN server thread failed: ...` 退出，**不挂起**；无重连/故障转移 |
| 客户端二进制误当服务端用 | 修复后直接报错退出（见 5.5） |

---

## 5. 验证中发现并修复的缺陷

### 5.1 静默错误：NCHW 客户端 + NHWC-only 服务端的转置方向写死

`distserver.cpp` 在需要转置时调用 `transposeTensor(..., /*srcLayoutIsNHWC=*/ true)`，把方向写死为"源是 NHWC"。当客户端配置 `inputsUseNHWC = false`（NCHW 输入）而服务端后端只支持 NHWC（Eigen 就是，`createComputeHandle` 会先抛异常再回退 NHWC）时，服务端把 NCHW 数据当 NHWC 转置，**输入特征整体错位，但连接正常、返回 0，搜索继续跑**。

实测（修复前）：同一配置下 `kata-raw-nn b all` 与本地参考**不一致**，客户端 rc=0——典型的静默数据损坏。修复：把第 5 个参数改为 `clientUseNHWC`。修复后逐字节一致，已加入回归用例。

### 5.2 `benchmarknn` 未初始化后端会话，分布式客户端在 Windows 上不可用

`benchmarknn.cpp` 没有调用 `Setup::initializeSession(cfg)`，因此 `DistProto::initNetworking()`（WSAStartup）从未执行，`benchmarknn` 走分布式后端时报 `could not resolve host ... WSAStartup 未执行`。已补上该调用（对其它后端也是正确的：CUDA 等同样需要一次性初始化）。

### 5.3 batch 上限不匹配只在运行中断链

客户端声明的 `nnMaxBatchSize` 大于服务端 `-max-batch` 时，握手照样成功，直到某个 batch 超限，服务端在请求中途抛错并断开连接——对局进行到一半才崩。现在 `serverHandshake()` 校验该值，超限即拒绝并提示"提高服务端 -max-batch 或降低客户端 nnMaxBatchSize"。

### 5.4 版本串把分布式后端显示为 dummy

`USE_BACKEND=DISTRIBUTED` 的构建在 `version` 里显示 `Using dummy backend`，benchmark 的 revision 也带 `-dummy`。已补 `USE_DISTRIBUTED_BACKEND` 分支，显示 `Using distributed inference client backend ...` / `-distributed`。

### 5.5 客户端构建可以运行 `distserver`（会困惑/递归）

`distserver` 子命令被编进所有构建，而 DISTRIBUTED 后端本身没有本地算力。现在该构建下直接报错："this binary ... cannot serve evals; build the server with the backend that owns the GPUs"。

上述修改涉及 `cpp/command/distserver.cpp`、`cpp/command/benchmarknn.cpp`、`cpp/neuralnet/distprotocol.{h,cpp}`、`cpp/main.cpp`，`build-dist` 与 `build-eigen` 均已重新编译并通过 26 项回归。

---

## 6. 性能结果

### 6.1 每 batch 传输开销（回环，同一次运行内对照）

| batch | 服务端纯计算 | 客户端往返 | 传输+序列化 |
|---|---|---|---|
| 1 | 62.5 ms | 63.9 ms | **1.4 ms**（2.1%） |
| 8 | 1156.5 ms | 1167.3 ms | **10.8 ms**（0.9%，含噪声） |
| 32 | 4187.5 ms | 4188.8 ms | **1.3 ms**（0.0%） |

多轮测量中该值在 −5 ms ~ +13 ms 之间波动（负值即测量噪声），**没有随 batch 单调增长的系统性趋势**。结论：回环链路上传输不是瓶颈；13 ms 可视为本机（CPU 已被 Eigen 占满）上的上界。

### 6.2 多机扩展（同连接数：分散到 N 台 vs 全部压 1 台）

为了让"多机"与"本机 CPU 上限"解耦，比较**相同连接数**下"分散到 N 台"与"全部压 1 台"（两侧占用核心数相同，若协议有串行化开销，前者会明显更慢）：

| 连接数 | 分散 N 台 | 全压 1 台 | 比值 | 本地同并发 Eigen |
|---|---|---|---|---|
| 1 | 14.62 evals/s | 16.34 | 0.89 | 18.45 |
| 2 | 19.98 | 21.60 | 0.92 | 36.70 |
| 3 | 25.05 | 26.67 | 0.94 | 50.65 |

- **分散/单机 = 89%~94%**：把同样的负载拆到多台机器上几乎无额外代价（6~11% 更可能是多进程各自加载一份模型导致 L3 不再共享，而不是协议开销）。
- **1→3 台吞吐 14.62 → 25.05 evals/s（×1.71）**，但同一台机器的本地 3 线程可达 50.65 evals/s。差距来自本机 CPU：只有 4 个 P-core，多进程调度会落到 E-core 上。证据是同一配置两次运行可差 1.7 倍（split 1 连接样本 [8.35, 14.62]；更早一轮出现过 2.8 倍），而本地同并发的两次运行几乎一致（18.45 / 18.41）。**因此本机数据不能用来断言"线性扩展"**——这是 CPU 替身的限制，不是传输层的结论。
- 可以确定的是：**每 batch 的传输开销是常数（1–11 ms）**，与服务器是 GPU 还是 CPU 无关；真实 GPU 场景下每台服务器各自独立计算，扩展上限由 GPU 数量决定，剩下的问题只是"传输常数 / GPU batch 时间"这一比例。

### 6.3 带宽（计数代理实测）

| 方向 | 每 eval |
|---|---|
| 客户端 → 服务端 | **31,878 B** |
| 服务端 → 客户端 | **1,490 B** |
| 合计 | **32.6 KB/eval** |

与协议公式吻合：`4 × (22×361 + 19) = 31,844 B` 上行；`4 × (361+1+3+6) ≈ 1,484 B` 下行（V7、19×19、不带 ownership）。

由此得到单机链路的上限（扣除协议开销前的理论上限）：

| 链路 | 可持续 evals/s 上限 |
|---|---|
| 1 GbE | ≈ 3.7k |
| 2.5 GbE | ≈ 9.4k |
| 10 GbE | ≈ 37.5k |

一个强 GPU 在小模型上很容易超过 3.7k evals/s，**1 GbE 会成为硬瓶颈**；batch=128 时单个 batch 就是 4.1 MB，1 GbE 光传输就要 33 ms。建议 GPU 集群用 10 GbE（或同机部署），至少 2.5 GbE。

### 6.4 RTT 敏感度（用户态代理注入时延）

单 batch 的传输开销 ≈ RTT + 处理时间（与 batch 大小无关，因为按 batch 计费）：

| 注入 RTT | batch=1 传输开销 | batch=8 传输开销 |
|---|---|---|
| 0 ms（回环） | −1.8 ms | 6.0 ms |
| 2 ms | 10.1 ms | 23.6 ms |
| 10 ms | 30.7 ms | 31.6 ms |
| 40 ms | 51.0 ms | 63.6 ms |

（代理对每个 TCP chunk 都加一次单向延迟，故实测值是真实链路的上界；趋势明确：**每次 RTT 都按 batch 支付**。）

换算成吞吐损失：batch=1、计算 62 ms 时，40 ms RTT 把单批时间拉长到约 113 ms，单连接吞吐约 **−45%**；batch=8、计算约 1.1 s 时，同样的 RTT 只损失约 **5%**（实测 evals/s 的绝对波动更大，因为同一次运行里服务端计算时间也会因 CPU 调度而抖动，故以"传输列"换算为准）。**batch 越大越能摊薄 RTT**，这与"引擎和 GPU 机器应同局域网"的部署建议一致。

---

## 7. 未验证项与风险清单

| # | 项 | 影响 | 建议 |
|---|---|---|---|
| 1 | **真实 CUDA 多卡未运行**（本机无 NVIDIA GPU） | 多卡吞吐、显存占用、P2P 行为均未实测 | 在一台 2 卡机器上做 2 卡 × 1 进程 vs 2 进程 × 1 卡对照，见第 8 节 |
| 2 | 服务端无配置文件 | `cudaShareModelWeights`、`openclTunerFile`、`useFP16`、TensorRT plan cache 等后端选项在服务端无法设置 | 给 `distserver` 加 `-config`，把 `ConfigParser` 传给 `createComputeContext` |
| 3 | 每连接一份模型权重 | N 条连接 = N 份显存；1 张卡挂 4 条连接可能 OOM | 每 GPU 限制 1–2 条连接；或启用 `cudaShareModelWeights` |
| 4 | 未做 transformer warmup | b18/b28 nbt 模型首次遇到新 batch size 时编译 cuDNN 计划，首帧慢；且失去 warmup 期间的 SDPA 宽松回退 | 在 `distserver` 里复用 `maybeWarmupComputeHandle` 的逻辑 |
| 5 | 无重连/故障转移 | 任一服务器掉线 = 引擎整体崩溃（当前是有意的 fail-fast） | 若要在线上使用，需要连接重建 + 重试队列；至少配合进程托管重启 |
| 6 | 无 TLS/鉴权 | 任何能访问端口的人都能消耗 GPU、读取评估结果 | 绑定内网、防火墙、SSH/WireGuard 隧道 |
| 7 | 无压缩 | 32.6 KB/eval 全量发送；高吞吐场景受链路限制 | 可考虑 fp16 传输或按 batch 压缩，但会引入 CPU 开销 |
| 8 | 连接→服务器映射在启动时固定 | 扩容/缩容必须重启引擎 | 后续可做动态服务发现 |
| 9 | 服务端默认 `-timeout 0` | 客户端异常消失（无 FIN）时连接线程永久阻塞 | 部署时显式设置 `-timeout 300` |

---

## 8. 在真实多卡机器上的下一步验证（建议照做）

**准备**：一台 2×NVIDIA GPU 的机器（GPU 记为 0、1），同一个 model 文件；另准备一台"引擎机"（可与 GPU 机同机，但要分别测"同机"和"跨机"两轮）。

```bash
# 1) 服务端（GPU 机）：两个进程，各占一张卡，便于看单卡吞吐
katago distserver -model model.bin.gz -listen 0.0.0.0:17731 -gpus 0 -max-batch 256 -timeout 300
katago distserver -model model.bin.gz -listen 0.0.0.0:17732 -gpus 1 -max-batch 256 -timeout 300

# 2) 客户端（引擎机）配置
#    distServers = gpu-host:17731,gpu-host:17732
#    numNNServerThreadsPerModel = 2
#    nnMaxBatchSize = 32

# 3) 正确性：远端 vs 本地同模型
katago gtp -model model.bin.gz -config dist.cfg      # 走远端
katago gtp -model model.bin.gz -config local.cfg     # 本地直连 2 卡
#    两边都执行 kata-raw-nn b all + 固定种子的 genmove，逐字节/逐手比较

# 4) 吞吐：1 卡 vs 2 卡 vs 2 机
katago benchmarknn -model model.bin.gz -config dist.cfg -batch-size 32 -iterations 50 -json
#    对比：本地 1 卡 / 本地 2 卡（numNNServerThreadsPerModel=2 + 设备0,1）/ 2 机各 1 卡

# 5) 关注
#    - 2 卡是否接近 2×（扩展效率），显存是否 2×（每连接一份权重）
#    - nvidia-smi 看两张卡的利用率是否都起来
#    - 跨机时用 iperf3 确认 ≥2.5 GbE，按 32.6 KB/eval 估算链路是否够
```

判定标准建议：2 卡相对 1 卡的吞吐 ≥1.7×；跨机（同局域网）相对同机下降 ≤10%；raw-NN 逐字节一致；`nvidia-smi` 两张卡均有负载。

---

## 9. 复现方式

```powershell
# 全量（约 6 分钟，26 项检查）
python dist_test\feasibility\harness.py

# 分组
python dist_test\feasibility\harness.py --groups correctness,robustness,gpu,perf
python dist_test\feasibility\harness.py --groups perf
```

脚本会自行拉起/回收 `distserver` 进程、生成配置（`out/cfg/`）、保留服务端日志（`out/logs/`），最终写入 `out/results.json`。端口从 17800 起自动挑选空闲端口。

---

## 10. 附：部署配置建议（当前实现）

```ini
# 客户端
distServers = gpu1:17731,gpu2:17731
numNNServerThreadsPerModel = 4      # 每台机器 2 条连接（2 台机器共 4 条）
nnMaxBatchSize = 32                 # 每连接 32，总 batch 可以到 64
distTimeoutSec = 600
```

```bash
# 服务端（每台机器一个进程；-max-batch 必须 >= 客户端 nnMaxBatchSize）
katago distserver -model model.bin.gz -listen 0.0.0.0:17731 \
                  -gpus 0,1 -max-batch 256 -boardsize 19 -timeout 300
```

用户文档见 `docs/DistributedInference.md`。

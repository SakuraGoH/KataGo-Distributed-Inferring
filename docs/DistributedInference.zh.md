# KataGo 分布式推理：验证报告（中文）

> 对应上游 PR：[lightvector/KataGo#1288](https://github.com/lightvector/KataGo/pull/1288)
> 日期：2026-11-26　引擎：KataGo 1.18.2 + 分布式后端（协议 v2）

---

## 1. 结论速览

| 问题 | 结论 |
|---|---|
| 分布式后端本身有没有性能问题？ | **没有。** 同机双卡、大模型下达到直算的 **98.8–99.6%** |
| 跨公网为什么慢了 55 倍？ | **链路带宽 + RTT**，可逐项算清（32.6 KB/eval、上行 ~10 MB/s、RTT ~24 ms） |
| 多卡调度路径可用吗？ | **可用。** `-gpus 0,1` 轮转正确，单进程多卡与每卡一进程行为一致（差 1–2%） |
| 数值正确吗？ | **正确。** 相同精度下与同机直算逐位一致 |
| 发现的实际缺陷 | 服务端**没有配置文件**，无法指定 `useFP16` 等后端选项 |

---

## 2. 测试环境

### 单机双卡（主实验）

- 1 台机器，2 × NVIDIA RTX 4080 SUPER 32 GB，128 核，CUDA 13.0 + cuDNN 9；
- 代码：上游 `master`（d91ea85）上应用本 PR 的补丁，从零编译：
  `build-cuda` 91 秒、`build-dist` 29 秒（`-j48`）；
- 两个模型：
  - 小模型 `sakura-b12c192`（7.9 M 参数卷积网络）——首批验证用；
  - 大模型 `kata1-tf3-b11c768`（**70.4 M 参数 transformer**）——本轮主测。

### 两台单卡（跨公网，用于对比）

- 2 台各 1 × RTX 4080 SUPER 的实例，跨公网，RTT 24–32 ms，实测上行 ~10 MB/s；
- 每台起一个 `distserver`，本地各开一条 SSH 隧道后聚合。

---

## 3. 单机双卡结果

### 3.1 大模型 b11c768（70.4 M transformer）

| 配置 | batch 8 (evals/s) | batch 32 (evals/s) |
|---|---|---|
| 直算，2 线程（每卡 1 线程） | 658 | 708 |
| 直算，4 线程（每卡 2 线程） | 742 | 733 |
| **A**：单进程 `-gpus 0,1`，2 连接 | 637 | 695 |
| **A**：单进程 `-gpus 0,1`，4 连接 | **734（98.8%）** | **730（99.6%）** |
| **B**：两进程各 1 卡，2 连接 | 642 | 694 |
| **B**：两进程各 1 卡，4 连接 | **737（99.2%）** | **729（99.4%）** |

- 每批传输开销 **~0.3 ms**，而前向计算是 **43–174 ms**——所以大模型下几乎看不出损失；
- GPU 利用率 **60–70%**，与直算相同（说明这个利用率是模型/批大小决定的，不是分布式造成的）；
- 连接分配日志确认轮转：`connection 0 (gpu 0)`、`connection 1 (gpu 1)`、`connection 2 (gpu 0)`…；
- 两进程方案下，两个服务端各记录 **102 / 102** 个 eval batch，确认两卡都在工作。

### 3.2 小模型 b12c192（7.9 M 卷积）

| 配置 | batch 8 | batch 32 |
|---|---|---|
| 直算，2 线程 | 5737 | 10378 |
| 直算，4 线程 | 10267 | 12304 |
| A，2 连接 | 5001（87.2%） | 7838（75.5%） |
| A，4 连接 | 9292（90.5%） | 10312（83.8%） |
| B，4 连接 | 9124（88.9%） | 10279（83.5%） |

小模型每批只有 2.5–6 ms，序列化的 0.3–1.7 ms 就占了 10–16%——**这正好说明损失来自“每批固定开销”，而不是分布式本身慢**。

### 3.3 精度问题（服务端无配置）

| 运行方式 | whiteWin | whiteLead |
|---|---|---|
| 直算，FP32（`useFP16 = false`） | 0.423235 | -0.235 |
| 直算，FP16（auto / true） | 0.423887 | -0.237 |
| 分布式（服务端默认 auto → FP16） | 0.423887 | -0.237 |

服务端没有配置文件，只能用后端默认精度；客户端即使写了 `useFP16 = false`，拿回来的仍是服务端的 FP16 结果（差 6.5e-4）。**相同精度下逐位一致**，所以这不是协议错误，而是服务端缺少配置入口。

---

## 4. 跨公网损失分解

以 b12c192、batch 8 为例：GPU 直算 3127 evals/s → 双机 4 连接 113 evals/s（约 28 倍）。

| 环节 | 倍数 | 数值依据 |
|---|---|---|
| 链路带宽 | ~10× | 每 eval 上行 32.6 KB；跑满需 102 MB/s，实测上行 ~10 MB/s |
| 隧道实现 | ~4× | SakuraLink（paramiko）~2.4 MB/s；`ssh -L` ~10–13 MB/s |
| RTT | ~1.2× | 每批一次往返，24 ms；batch 8 传输 107 ms |

结论：**这部分损失是物理链路和隧道实现决定的，与引擎无关。**同机/局域网部署时这部分直接消失（见第 3 节）。

---

## 5. 当前限制（与 PR 中一致）

1. **跨机受网络支配**：上行 32.6 KB/eval，24 ms RTT 下单连接上限约 150–300 evals/s；同机或 10GbE 才能吃满 GPU。
2. **服务端没有配置文件**：无法设置 `useFP16`、`cudaShareModelWeights`、`openclTunerFile` 等；客户端与服务端精度不一致时数值会有 1e-4 级差异（见 3.3）。
3. **协议 v2 不向下兼容**：客户端与服务端必须同版本重编。
4. **每连接一份模型权重**：单卡连接数过多会吃显存。
5. **无重连/无故障切换**：服务端挂掉，引擎直接中止。
6. **无认证/无 TLS**：预期部署是服务端只绑回环 + SSH 隧道。
7. **未验证同局域网与真实大模型跨机**：跨机只用过小模型，大模型只在单机双卡上测过。

---

## 6. 建议的下一步

1. 给 `distserver` 加 `-config`（至少支持 `useFP16`），或在握手协议里带上精度；
2. 用 b11c768 做一次跨机测试（1GbE 或 10GbE），验证大模型下跨机损失是否显著缩小；
3. 给分布式路径加一个可重复的基准脚本，避免以后回归。

---

## 7. 复现命令（单机双卡）

```bash
# 编译
cmake -S . -B build-cuda -DUSE_BACKEND=CUDA -DNO_GIT_REVISION=1
cmake --build build-cuda -j48
cmake -S . -B build-dist -DUSE_BACKEND=DISTRIBUTED -DNO_GIT_REVISION=1
cmake --build build-dist -j48

# 变体 A：单进程管两张卡
./build-cuda/katago distserver -model model.bin.gz -listen 127.0.0.1:17731 -gpus 0,1 -max-batch 256
# 变体 B：每卡一个进程
./build-cuda/katago distserver -model model.bin.gz -listen 127.0.0.1:17731 -gpus 0 -max-batch 256
./build-cuda/katago distserver -model model.bin.gz -listen 127.0.0.1:17732 -gpus 1 -max-batch 256

# 客户端（build-dist）
# distServers = 127.0.0.1:17731            # A
# distServers = 127.0.0.1:17731,127.0.0.1:17732   # B
# numNNServerThreadsPerModel = 4
./build-dist/katago benchmarknn -model model.bin.gz -config dist.cfg -batch-size 32 -iterations 10 -json
```

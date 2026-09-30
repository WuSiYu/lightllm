# 260905-FlexTP 现状速记

## 调度器

- `fixed_tp2`：仅两个独立 TP2，所有请求固定 TP2。
- `fixed_tp4`：仅一个 TP4，所有请求固定 TP4。
- `naive`：按 4000 token 硬阈值选 TP2/TP4，允许 MPS 重叠，不做智能调度。
- `v2`：旧版 FlexTP 重写/兼容版，按旧接口与负载逻辑工作。
- `v3`：deadline、lease、bundle、token-credit 的基础版本。
- `v4`：短长请求严格隔离。
- `v5`：在候选级加入 MPS-first 可行性控制。
- `v6`：naive 路由基础上按 SLO 效用决定是否改变默认分流。
- `v7`：在线冲突价格与 work-conserving 调度。
- `v8`：固定 epoch、双 lane quota/wavefront。
- `v9`：MLFQ aging + CFS virtual runtime，借鉴 OS 调度。
- `v10`：EEVDF/deadline 排序与 overlap-slack reservation，保留阈值路由。
- `v11`：v10 的连续 TP 选择，取消长度硬阈值并加入 TP footprint price。
- `v12`：v6 生命周期上的预测完成时间 + footprint price，TP4 有 service-ratio guard。

## 运行口径

- 当前可用路径：普通 PD；NIXL 模式暂不作为可用基线。
- 旧调度器启动脚本参考：`start-cluster3_70b_p22.4d4_mps_flex_v2.sh`。
- 新调度器启动脚本：`start-cluster4_70b_p22.4d4_mps_flex_v[3-12].sh`
- 生产拓扑：TP2(0,1)+TP2(2,3)+TP4(0,1,2,3)；MPS 共享 GPU。decode占用gpu 4-7，与本地prefill配对，不是优化范围。

## 测试

- 模拟器：`test/benchmark/service/flex_tp_step_sim.py`；离散 step，synthetic-5pct/ServeGen mm-image，decode 可视为无瓶颈，支持 rate/TTFT/SLO sweep。
- 更大规模算法对比目前以模拟器结果为主；GPU 长流量实测尚未完成。

## 约束

- 新的文档和文件夹标题使用 `YYMMDD-...` 日期标记。

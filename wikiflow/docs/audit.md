# 审阅记录：2026-10-05

本轮只整理已汇报身份修正后的 binary / relative continuous，保留冻结数学定义。源实验目录和共享缓存只读；在独立 clone 上工作。仓库原 main 只有 README，所有实验账本均未改写。

## 核对项

- `identity_policy.py`/manifest v1：block 79745475，同名导演历史不 join；Naevis review-only 保留。身份策略的公开副本只保留必要规则及原 SHA。
- `preflight.py`/`fit26.py`：复核 Log13 joint-T 第13列、mean BCE 的 L2=.01、Newton 初始化/停止、HGB12 固定参数和 own-TRAIN minleaf。
- `training_contract.py`/`run_authorized26.py`：复核 relative response `gR=log1p(max(0,actual/T−1))`，Ridge alpha=Nλ，所有模型与 scaler 只取成熟 TRAIN；无编辑训练过滤。
- `pilot_math.py`/`relative_contract.py`/两个 corrected evaluator：未知整月 NA、零事件月份支持、tie、macro/micro、full-list AP、graded gain 一次变换。
- `rolling_training_support_zero_fit/verify_support_no_fit.py` 与 v2 `prepare_v2_no_fit.py`：按月长日均、七个完整公布月、两重 cold gate、两完整月 revision parent 净字节 max/sum 阈值。
- acquisition/daily parser：字面 TSV、全部内部来源、count10 边界、canonical-only、显式 PV0/缺日期、覆盖与 parent/hidden 守门。新采集脚本保留 raw，不使用原脚本的自动删除选项。

原程序依赖深层 sessions 和硬编码机器路径，且会更新共享累计 ledger，无法作为安全复现入口直接运行。本包提取纯计算/模型代码，把路径换成便携目录，失败停止；身份、数学、候选和模型配方保持一致。源码 hashes 和必要派生工件 hashes 在 `data/provenance.json`。

## 时间与选择的精确边界

每个 M 月只用 target≤M−1 的已知标签；自身行特征≤target−1，预测特征≤M−1。历史名单/标题和 revised vintage 为当前资料回溯应用；未核实原发布时延，因此立即成熟是操作假设。

binary 继承的早期容量控制：

| fold | TRAIN | INNER评价 | TRAIN N/P |
|---|---|---|---:|
|F1|2023-10—2023-12|2024-01—2024-04|649/40|
|F2|2023-10—2024-04|2024-05—2024-08|1590/71|

四个固定候选为 Log12 λ=.1/.01、HGB stump/depth2。8 个 INNER 月等权 binary NDCG50 选择 Log12 .01（.155725）；depth2 HGB（.110317）是保留的固定比较方法，**不是早期赢家**。后来 v2 标签、joint-T Log13 特征与规则/主模型选择看过历史诊断，不能声称端到端选择只接触未见早期资料。数值参数保持，身份修复和本轮无新调参。

relative λ=.01/.1 只用 2024-02/05/08 三个早期历史边界，以等月 graded NDCG50 选 .01；HGB固定。其既有强规则也不能据回测均值声称独立胜出。

## 复现与限制

实际重训覆盖 24 月 × 4 家族。逐行分数、96 个完整月排序、所有 K=5/10/20/50/100/200 的 2,160 个方法/月/K 记录、6,480 个 NDCG/AP/TP 字段核验。另从 48 月原始缓存在新临时输出构建 11,080 行/5,388 候选，身份/标签/资格/阈值精确一致，浮点特征误差≤4.55e−13。详细证据在 `results/`，带实际拟合数、依赖、seed、threads、耗时和容限。

没有发现影响已报指标的实质 bug，也没有替换已报数字。修正了新包的候选序列化排序：history=target/article，evaluation=target/pageid，与权威视图一致；这不改变原实验。

本轮不全量联网重取数据，不运行新调参，不读取 September 2026，不执行 bootstrap 重估。离线 fixtures 验证网络响应解析；不能把这称为 fresh API 下载成功。缺历史 identity certificate、API vintage变化、发布时延、编辑净字节代理、非纯 AI 维护范围、suppressed clickstream 和反复开发选择，均限制外推。独立测试/实际部署收益仍未验证。

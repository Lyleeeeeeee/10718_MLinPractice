# WikiFlow：身份修正后的最小复现包

在 WikiProject Artificial Intelligence 的维护名单中，按月推荐当前未升温、近期大幅编辑较少、下月可能出现内部导航升温的文章。名单固定为 1,187 个 canonical article；它包含人物、作品和其他维护范围文章，不是纯 AI 技术集合。编辑历史是候选资格代理，不是内容质量或维护必要性的标签。

本包只整理 2026-10-04 身份修正后的 binary 与最新 **relative continuous** 主线。旧 absolute continuous、诊断融合、新渠道/来源实验、完整 sessions、模型 pickle、原始大数据均未纳入。

[设计文档](docs/design.md) 面向课程队友，逐项说明月度人工检查用途、数据与时间合同、原13特征、模型选择、同池评价及未完成事项；[审阅记录](docs/audit.md) 保留核对证据。

## 快速复现：真的重训

从仓库根目录运行。推荐 Python **3.13.5**；实际验证环境为 NumPy 2.3.3、SciPy 1.16.2、scikit-learn 1.7.2、threadpoolctl 3.6.0。其他支持 Python 版本可运行，但位级浮点复现需使用记录的环境。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r wikiflow/requirements.txt
export PYTHONPATH=wikiflow/src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
python -m unittest discover -s wikiflow/tests -v
python -m wikiflow.reproduce --mode refit --output wikiflow/runs/my-refit
```

`refit` 对 2024-09—2026-08 的每月执行 Log13、binary HGB12、relative Ridge12、relative HGB12：**24×4=96 次真实拟合**，固定参数，无新搜索。输入和每折 TRAIN keys 校验后，由各折 TRAIN 重算 imputation/scaler；与历史基准逐行比较模型分数、逐月完整排序以及所有 6 个 K 的指标。任何不一致超过容限即停止。每次运行须使用新的输出目录。

```bash
python -m wikiflow.reproduce --mode replay --output wikiflow/runs/my-replay
```

`replay` 只重算已存预测的指标，实际拟合数为 0，与 `refit` 清楚区分。输出含 `fit_trace.json`、封存预测、逐月/汇总 CSV、`verification.json`。提交的 results 是 push 前实际 refit 的证据，不把 replay 称为重训。本次先对冻结快照重训96次，再验证完整缓存→清洗→重训96次，合计192次固定拟合，无调参；两条路径所有逐月排序与已报指标一致，各约10.9秒。

## 已复现的 K=50 结果

候选池共 **5,388 个 article-month**。24 个历史月中，23 月全部标签已知，22 月含事件；自然事件 182，Top50 槽位 1,150。2026-04 有未知标签，整月指标 NA；2025-04 无事件，TP/precision 计入，NDCG/AP/recall 为 NA。2026-07 移除身份未知候选后恢复完整评价。

| Binary 排序方法 | macro NDCG50 | macro AP（完整列表） | TP50累计 |
|---|---:|---:|---:|
|volume|0.123384165982|0.062143794623|45|
|strong CS|0.262764324285|0.133360599203|70|
|Log13|0.261323643627|0.127705211410|75|
|HGB12|0.254520450786|0.114735066542|77|

| Relative 排序方法 | macro graded NDCG50 | TP50累计 |
|---|---:|---:|
|volume|0.092810605545|45|
|Ridge12|0.138377139655|63|
|HGB12|0.127657749963|57|
|PV trend/T|0.184767834660|77|

[完整主方法表](results/summary_at50.csv) 同时保留 binary 7 方法、relative 8 方法；[逐月表](results/monthly_at50.csv) 与原结果核对。HGB12 指 **12 列特征**；原目录名的 HGB26/authorized26 指身份修复的 **26 次拟合**，不是 26 个特征。

无稳定 ML 优于强 CS/PV 规则的证据，均值胜过 volume 不能改写为稳定胜出。所有历史月份反复研究，叫开发回测，不叫独立测试。原研究区间为描述性区间，不构成独立验证；本最小包复现点预测和指标，没有再跑 bootstrap 或调参搜索。

## 数据、时间与身份

公共来源是 [Wikimedia monthly Clickstream](https://dumps.wikimedia.org/other/clickstream/readme.html)、[article daily pageviews](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html)、[MediaWiki revisions API](https://www.mediawiki.org/wiki/API:Revisions) 和当前 WikiProject talk category。

Clickstream 按所有 Wikipedia 内部 `link` 来源→canonical destination 的公布计数求和；不是独立读者数、搜索曝光或全部 pageviews。清洗保留字面 TSV 的四列、published count=10 边界、任意内部来源；不限定来源也属于 AI 名单。未公布边受到抑制；缺失 article/month 是未知，不能补零。原采集用 redirect/normalized mapping 确定 canonical 维护名单，但**没有另行把 redirect alias 的 destination 计数聚合入 canonical**；本包保持原计数定义。

固定历史输入范围为 2022-09—2026-08，模型目标 2023-10—2026-08，开发评价目标 2024-09—2026-08；用于预测 M 的 CS/PV 特征截至 M−1，PV 截止最终 2026-07-31。编辑元数据截至 2026-08-01 exclusive。标签取目标月已经公布的 incoming article count。2026-09 holdout 不在派生快照里，入口拒绝该月及之后。

历史采用当前 2026 名单、当前标题映射和修订后的数据 vintage，**并假设 M−1 月 CS/标签在 M 月初已成熟可用**。真实历史 release latency 未还原，不是已经证明的月初生产可部署方案。

权威身份修正是 `identity_correction_v1_Daniel_homonym_quarantine`，[公开策略](config/identity_policy.json) 保留源 manifest SHA256。pageid **79745475** 的 Daniel Kokotajlo 当前同名槽身份不明：训练、候选和标签统一阻断，不把导演 pageid 62422990 的历史转移给它，不制造零标签。历史已有排除 1164/72417803 保持；Naevis 78522383 的 review-only 警示保留，不自动删行。当前映射并不认证所有文章的历史身份，特别是 M−12 季节特征。

## 公式与候选

令目标月为 M，当前完整月 c=M−1，月长为 D，文章每月公布的内部 incoming count 为 I，日均 v=I/D。

对结束于 c 的 6 个完整月日均，m=median(v)，s=1.4826×median(|v−m|)：

```text
U_M = max(200, D_M*m + max(100, 0.5*D_M*m, 3*D_M*s))
x_M = I_c * D_M / D_c
T_M = max(U_M, 1.25*x_M)
binary event = 1[actual_M > T_M]                 # strict >
R_M = max(0, actual_M / T_M - 1)
gR_M = log1p(R_M)                               # once
```

U 的 floor 和 padding 是固定政策，不是校准过的“三倍标准差”概率区间。binary 等价于 actual>U **且**按日环比严格>25%；不是预测单纯未来流量。旧 absolute `max(0,actual−T)` 不在该 continuous 标签中。

同一 candidate pool 满足：固定身份有效、当前 count≥100、当前及前 6 月 incoming 均有公布记录、质量/编辑覆盖完整；当前 rate≤U_M/D_M，且当前 count≤用此前 6 月（不含当前）冻结的 U_c；最近两完整月 `[M−2月初,M月初)` 内非 minor 单次**净字节绝对变化**<500，累计绝对变化<1000。净变化按每次 revision 与真实 parent 的 size 差计算，回退也计入，minor 忽略；这是净字节代理，无法发现字节相抵的大幅改写。

**训练保留质量完整且 current-cold 的 expanded pool，不按编辑候选过滤。** 每折只用 target<M、标签已知的行；未来 M+ 标签、未知历史标签不得进入 TRAIN。派生 pool 共 11,080 行。

## 特征、预处理与固定模型

|index|原12列|
|---:|---|
|0/1|log1p 当前/前一月 CS 日均|
|2/3|log1p 最近3月均值/最近6月中位 CS 日均|
|4|最近6月日均 OLS slope，分母17.5|
|5/6|log1p 目标 M−12 月日均；缺失则 current fallback + missing flag|
|7/8/9|log1p 当前整月 PV 日均；最近14天/前14天 log1p 均值之差；PV missing flag|
|10/11|log1p robust daily sigma；log1p(x_M)−log1p(U_M)|

Log13 再加第13列 `log1p(x_M)−log1p(T_M)`。PV 整月不完整时相关量保留 None；以每折成熟 TRAIN median 插补、计算均值和**总体**标准差，std<1e−12→1。missing flag 与“无观测列 median=0 的计算性 fallback”保留，不把未知 outcome 当负例。树也使用相同 TRAIN 标准化输入，保持历史配方。

Log13 用 row-mean BCE + 0.5×.01×||β||²，截距不惩罚，无 class/month weighting，零 slopes+TRAIN prevalence logit 初始化，固定 damped Newton，gradient≤1e−9，最多200轮/60次线搜索，无重试。Ridge 对 gR 最小化 mean squared error +.01×||β||²，因此 sklearn `alpha=N*.01`，cholesky、有截距。

两个 HGB 均 30 轮、depth2、leaves4、bins32、lr=.03、L2=1、minleaf=max(50,ceil(.1N))、seed718；无 early stopping、validation_fraction、warm start、class weight。binary loss=log_loss；relative loss=squared_error。月折分别从头 expanding fit。

参数和目标选择过程须区分：binary 数值配方继承早期 TRAIN 内控制实验（F1训练2023-10—12/评价2024-01—04，F2训练截至2024-04/评价2024-05—08；Log12 .01为赢家，HGBdepth2是固定比较），后来的 joint-T 特征与规则/主模型叙事参考过已见开发诊断，不能声称整个选择从未看开发结果。身份修复固定参数、无新搜索。relative Ridge λ 在 **2024-02/05/08** 三个早期历史折，只比较 .01/.1 的等月 graded NDCG50，选择 .01；平局≤1e−12 选更强正则 .1。HGB 固定，没有树网格搜索。[冻结早期选择](config/early_selection.json) 及选择函数/测试随包提供，此次没有重新搜索。旧 split 名称 validation/dev 不恢复独立性。

## 规则与指标

所有方法对相同候选评分，score 降序→当前 count 降序→pageid 升序。volume=当前 count；strong CS=`log1p(x_M)−log1p(T_M)`；momentum 用当前/前月日均 ratio 延续，relative 另保留 ratio clip[.5,2] 的固定版本。PV forecast=`x_M*exp(PV14logtrend)`，PV 不完整时 fallback=x_M；分别 against U/T 作 log margin。

浮点 tie 必须可追溯：每次独立重构规则公式，误差≤1e−12 后保留原 canonical 规则分数，与原 corrected evaluators 一致；这是规则浮点/tie 处理，不用于代替 ML 重训。四个 ML 全部重训、每月完整排序精确相同。

DCG=Σgain/log2(rank+1)。binary gain=event；relative 主 gain=gR，直接 identity gain：**没有第二次 log，没有 2^gain−1**。NDCG 除以该月理想排序 DCG。AP 使用完整 candidate ranking 的 rank-tiebroken average precision，不是截断 AP@50。macro 指各有效月等权；TP/slots/events 累计，micro P/R 和 pooled gR capture 按对应总量计算，不能与 macro 混用。未知候选不删、不补、不 refill；任一 future label unknown 则整月 NA；P=0 月 precision/TP=0，其 NDCG/AP/recall NA。

## 从公共源下载与从缓存清洗

默认重训**不联网**，使用 `data/panel.json.gz`（约2MB派生快照）与对应校验和。`data/provenance.json` 保存便携相对源路径、原工件 SHA256 与包输入 SHA256；`public_download_provenance.json` 是公共 dump 的 URL/校验记录。派生数据包含公共文章名称/pageid和统计量，不含编辑者、revision 正文、评论、凭据或机器路径。

以下命令在全新 `wikiflow/local_data/` 下采集，保留 raw/cache，不替换已存在输出。重新获取的当前名单/API历史可能与冻结 vintage 不同，不能自动宣称精确重现旧实验。复现旧 scope 时用已提供的固定 `data/title_mapping.csv`。

```bash
mkdir -p wikiflow/local_data
python -m wikiflow.acquisition topic --output wikiflow/local_data/new_mapping.csv
python -m wikiflow.acquisition clickstream --month 2024-08 \
  --mapping wikiflow/data/title_mapping.csv \
  --raw wikiflow/local_data/clickstream-enwiki-2024-08.tsv.gz \
  --output wikiflow/local_data/incoming_2024-08.csv.gz
python -m wikiflow.acquisition pageviews --mapping wikiflow/data/title_mapping.csv \
  --start 2022-09-01 --end 2026-07-31 --output wikiflow/local_data/daily.csv.gz
python -m wikiflow.acquisition revisions --mapping wikiflow/data/title_mapping.csv \
  --start 2023-08-01 --end-exclusive 2026-08-01 \
  --output wikiflow/local_data/revisions.json --coverage wikiflow/local_data/coverage.json
```

完整 raw 重建需逐月取得 **2022-09—2026-08** 的 `incoming_YYYY-MM.csv.gz`，耗费较大，不是 quickstart。本次没有重新下载。使用已有缓存的清洗命令：

```bash
python -m wikiflow.preparation --mapping wikiflow/data/title_mapping.csv \
  --incoming wikiflow/local_data --pageviews wikiflow/local_data/daily.csv.gz \
  --revisions wikiflow/local_data/revisions.json --coverage wikiflow/local_data/coverage.json \
  --output wikiflow/local_data/rebuilt_panel.json.gz
```

清洗结果可直接接入同一固定训练/核验入口（必须匹配冻结 benchmark 的 keys、label、资格与数值容限，否则先停止诊断）：

```bash
python -m wikiflow.reproduce --mode refit \
  --prepared-panel wikiflow/local_data/rebuilt_panel.json.gz \
  --output wikiflow/runs/my-cached-refit
```

采集 revision ID/parent/timestamp/size/minor/SHA 和覆盖区间，检查 parent 连续性；不请求用户名、评论、正文。缺覆盖/parent/隐去记录则 quality unknown，不能造零编辑。PV 区分 API 显式0和缺日期。所有下载及清洗拒绝 holdout；重新下载不认证历史 title-slot 同名身份，需沿用隔离策略和另行审查。

[缓存重建检查](results/cached_preparation_verification.json) 和公共合成 fixture 测试验证采集解析/清洗边界。历史特征形成时存在 NumPy/标准库两种浮点求和路径，新的原始缓存重建允许浮点容限；用于位级预测复现的冻结派生快照保持原浮点值。完整联网采集、实际发布时延认证、全目录历史身份认证和封存 September 测试仍未执行。

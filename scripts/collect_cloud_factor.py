"""超大规模云厂商（hyperscaler）季度景气因子采集器。

目的：为 DDOG（Datadog，云可观测性 SaaS）构造**需求侧行业景气代理变量**。
AMZN / MSFT / GOOGL 三家合计的收入与云基建资本开支，是公有云 / AI 基建投入的
公开、可复现、无前视的同业景气指标。

数据源：SEC EDGAR 公开 XBRL 接口（美国政府公开数据，官方允许程序化访问）
  https://data.sec.gov/api/xbrl/companyconcept/CIK{10位CIK}/us-gaap/{Tag}.json
  用 companyconcept 而非 companyfacts：单公司单 tag 的 JSON 仅几十 KB，
  三家 × 6 个 tag ≈ 16 个请求即可覆盖全部历史（companyfacts 每份 10–40 MB）。

产出：
  data/processed/cloud_complex_quarterly.csv   公司 × 季度的收入 / capex / 同比
  data/processed/cloud_complex_panel.csv       三家公司按季度聚合的合计面板
  data/raw/sec_cloud/<today>/*.json            每个 companyconcept JSON 原样快照

──── 口径与关键实现细节（决定数字能不能用） ────────────────────────────────

1. YTD 差分。现金流量表的 capex 在 10-Q 里是**年初至今(YTD)累计值**，必须差分出单季：
     Q1 = 3M；Q2 = 6M − 3M；Q3 = 9M − 6M；Q4 = 12M − 9M。
   实现不按"日历年"硬编码，而是按**同一 start 日的申报区间链（chain）**自动差分——
   MSFT 财年 7/1 起算，"年初"是 7 月，硬编码 1/1 会全错。
   差分合法性：本期区间天数 − 上期区间天数 ∈ [80,100] 才算相邻季度；链条缺环
   （例如缺 9M）时**不下推**，宁可缺值也不编数。

2. 重述（restatement）与"混搭口径"的坑。同一 (start, end) 在多次申报里可能有不同值
   （10-Q/A、10-K/A、后续年度的比较期重述）。若对 (start,end) 各自独立取"filed 最新"，
   会出现**用新口径的前一季 YTD 去减旧口径的本期 YTD**，得到完全错误的差额。
   实测反例（MSFT，SalesRevenueNet，2017Q2）：本期 FY2017=89.95B(2017-08-02 原报) −
   9M=70.97B(2018-04-26 ASC606 重述) = 18.98B，真值应为 23.32B。
   因此本脚本做两条硬约束：
     (a) **同一链内 prev 的 filed 必须 ≤ item 的 filed**（只能用"当时已知"的口径）；
     (b) 默认口径 VINTAGE="as_reported"（首次申报值，point-in-time），使整条序列口径一致；
         这种混搭在 MSFT ASC 606 切换点会造成约 +7% 的虚假同比跳变，故不作为默认。
   如需"重述后口径"，把 VINTAGE 改成 "latest" 即可；脚本会打印两种口径的差异清单。

3. tag 回退。收入优先 RevenueFromContractWithCustomerExcludingAssessedTax，缺失时回退
   Revenues / SalesRevenueNet；capex 优先 PaymentsToAcquirePropertyPlantAndEquipment，
   缺失时回退 PaymentsToAcquireProductiveAssets（AMZN 2017Q2 起换 tag，必须回退）。
   每个季度实际用到的 tag 记录在 tag_revenue / tag_capex 列。
   跨 tag 择值规则：取 first_filed 最早的候选（真正的 point-in-time），若落在同一申报
   窗口（±7 天）内则按上述优先级取 tag。

   另外，**只有当两个 tag 被证明为同一口径时才允许跨 tag 拼接 YTD 链条**
   （判据：重叠期取值差异 <= 0.5%；无单季重叠时退化到原始区间 fact 比较）。
   实测：GOOGL 的 Revenues ≡ RevenueFromContractWithCustomerExcludingAssessedTax
   （17 个重叠单季差异 0.00%）→ 允许拼接，于是 2020Q4/2021Q4 的 first_filed 从
   2023-02-03（该 tag 只在后续年报的比较期出现）提前到真实年报日 2021-02-03 / 2022-02-02；
   反之 AMZN 两个 capex tag 在 FY2016 差 15.84%（6.737B vs 7.804B）→ 判定口径不同，
   严格不混算（2017 年换 tag 时宁可留一个缺口，也不跨 tag 相减）。

4. 过滤：只用 form 形如 10-*（10-Q / 10-Q/A / 10-K / 10-K/A）的 fact，**排除 8-K**。

5. 季度标签按 period_end 的**自然日历季度**（2023-09-30 → 2023Q3）。三家都用自然月
   财季，故 MSFT 的 FQ1（9/30 结束）落在日历 Q3，跨公司可直接对齐。

6. first_filed = 构造该值所用 fact 中最大的 filed（该值首次可得的申报日），恒 ≥ period_end，
   无前视；latest_filed = 该期末所有候选 fact（全 tag、全口径）的最大 filed，用于观察重述。
   注意：SEC 的 XBRL 分阶段强制从 2009 年起，故 2007–2009 那几期的 first_filed 是
   "首次以 XBRL 形式可得"的日期（往往是后续年报的比较期），**晚于**原始纸质申报日——
   属于保守偏差（只会推迟可得性），不会引入前视。

──── 已知数据缺口（脚本运行时会打印，已在报告中说明） ────────────────────
  * AMZN capex 2017Q2 缺失：Amazon 在 2017 年把 capex 的 tag 从
    PaymentsToAcquirePropertyPlantAndEquipment 换成 PaymentsToAcquireProductiveAssets，
    新 tag 的 2017 年比较期只提供了 9M 与 FY、没有 H1，链条缺环 → 不下推（Q4 仍可由
    FY−9M 得到）。补该期需要跨 tag 相减（≈3.40B），本脚本不做，以免引入口径混算。
  * AMZN capex 口径在 FY2023 变化：tag 值 FY2017–FY2022 与公开引用的年度 capex 逐年完全
    一致（11.955/13.427/16.861/40.140/61.053/63.645B），FY2023 起等于"现金 PP&E + 当年融资
    租赁新增"（48.133+4.596=52.729；77.658+5.341=82.999）。**2023 年四个季度的同比因此偏高**。
  * GOOGL 收入/capex 起点为 2014Q3：2015 年以前的 10-Q 未按这两个 tag 打季度/YTD 标签，
    companyconcept 里没有 2013–2014H1 的可差分区间；capex 2014Q3 亦缺（无 H1 2014 可比）。
  * MSFT capex 缺 2007Q3–2008Q2：capex tag 最早的 fact 是 FY2008 年报（12M），此前没有链条。
  * MSFT 收入 2016Q3–2017Q2 用 SalesRevenueNet（原报口径，如 2017Q2 = 23.317B）；
    若改用重述口径（10-K FY2018 比较期）则为 25.605B（+9.8%）。默认取原报口径，
    否则 ASC 606 切换点会出现约 +7% 的虚假同比跳变。
  * MSFT 的 capex 不含融资租赁（FY2024 融资租赁另增约 $19B），
    与 AMZN tag 在 FY2023 后的口径不一致 → 跨公司 capex 合计存在口径噪声。
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_utils import LEGAL_BASIS, NotFound, get_json, save_raw, write_table  # noqa: E402

VINTAGE = "as_reported"          # "as_reported"（默认，point-in-time）| "latest"（重述后口径）
TIE_DAYS = 7                     # 跨 tag 择值：first_filed 在此窗口内视为"同一时点"，按优先级定

CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"

COMPANIES: list[tuple[str, str, str]] = [
    ("AMZN", "0001018724", "Amazon.com, Inc."),
    ("MSFT", "0000789019", "Microsoft Corporation"),
    ("GOOGL", "0001652044", "Alphabet Inc."),
]

# tag 优先级（高 → 低）。补充回退 tag 在个别年份是唯一可得口径：
#   AMZN 2016Q4 之前收入只有 SalesRevenueNet；MSFT 2011–2016 收入只有 SalesRevenueNet；
#   GOOGL 2025Q2 之后又切回 Revenues；AMZN capex 2017Q2 起换成 PaymentsToAcquireProductiveAssets。
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
]
CAPEX_TAGS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
]
METRIC_TAGS = {"revenue": REVENUE_TAGS, "capex": CAPEX_TAGS}

FORM_RE = re.compile(r"^10-")                     # 10-Q / 10-Q/A / 10-K / 10-K/A …
QMONTH = {"03": 1, "06": 2, "09": 3, "12": 4}
BUCKETS = {"q": (80, 100), "h": (170, 190), "n9": (260, 280), "fy": (355, 375)}


# ---------------------------------------------------------------- 抓取 --------
def _fetch_tag(ticker: str, cik: str, tag: str) -> dict | None:
    url = CONCEPT_URL.format(cik=cik, tag=tag)
    try:
        d = get_json(url, source="sec", note=f"{ticker} companyconcept {tag}")
    except NotFound:
        print(f"    - {ticker} {tag}: 该 tag 不存在（404），跳过")
        return None
    except Exception as e:
        print(f"    ! {ticker} {tag}: 抓取失败 {str(e)[:80]}")
        return None
    save_raw("sec_cloud", f"{ticker}_{tag}", d, url=url,
             note=f"{ticker} XBRL companyconcept {tag}")
    return d


def _usd_facts(doc: dict | None) -> list[dict]:
    return ((doc or {}).get("units") or {}).get("USD") or []


# --------------------------------------------------------- 区间整理 --------
def _bucket(days: int) -> str | None:
    for name, (lo, hi) in BUCKETS.items():
        if lo <= days <= hi:
            return name
    return None


def _is_quarter_end(period_end: str) -> bool:
    _, m, d = period_end.split("-")
    return m in QMONTH and int(d) >= 28


def _qlabel(period_end: str) -> str:
    y, m, _ = period_end.split("-")
    return f"{y}Q{QMONTH[m]}"


def _build_chains(facts: list[dict], tag: str) -> dict[str, dict[str, list[dict]]]:
    """{start: {end: [同一区间的多个申报版本（可来自不同 tag，若已证明等价），按 filed 升序]}}。

    只保留 form=10-* 且区间天数落在四个标准桶里的 fact；时点值（无 start）丢弃。
    """
    fx: dict[str, dict[str, list[dict]]] = {}
    for u in facts:
        form = str(u.get("form") or "")
        if not u.get("start") or not u.get("end") or u.get("val") is None:
            continue
        if not FORM_RE.match(form):
            continue                                  # 排除 8-K 等
        days = (date.fromisoformat(u["end"]) - date.fromisoformat(u["start"])).days
        b = _bucket(days)
        if b is None:
            continue                                  # 过渡期 / stub 区间
        fx.setdefault(u["start"], {}).setdefault(u["end"], []).append({
            "start": u["start"], "end": u["end"], "days": days, "bucket": b,
            "val": float(u["val"]), "filed": str(u.get("filed") or ""),
            "form": form, "accn": str(u.get("accn") or ""),
            "tag": str(u.get("_tag") or tag),
        })
    for s in fx:
        for e in fx[s]:
            fx[s][e].sort(key=lambda r: (r["filed"], r["accn"], r["tag"]))
    return fx


def _pick(versions: list[dict], vintage: str) -> dict:
    return versions[0] if vintage == "as_reported" else versions[-1]


def _quarterly(facts: list[dict], tag: str, vintage: str) -> tuple[dict[str, dict], list[dict]]:
    """单季序列 {quarter: 记录} + 内部一致性校验记录。核心是 YTD 差分。

    facts 可以来自多个**已证明等价**的 tag（见 _equivalence_groups）；
    记录里的 tag = 收盘 fact（本期区间）所用的 tag。
    """
    chains = _build_chains(facts, tag)
    out: dict[str, dict] = {}
    checks: list[dict] = []

    # 每个期末的所有候选（跨 tag 的 latest_filed 由调用方汇总，这里先给本 tag 的）
    per_end_latest: dict[str, str] = {}
    for ends in chains.values():
        for e, versions in ends.items():
            per_end_latest[e] = max([v["filed"] for v in versions] + [per_end_latest.get(e, "")])

    for start, ends in chains.items():
        chain_q: dict[str, tuple[str, float]] = {}      # end → (quarter, 单季值)，供链内一致性校验
        for end in sorted(ends):
            item = _pick(ends[end], vintage)
            val, prev, basis = None, None, ""

            # 找链条内紧邻的上一期（天数差 80–100 天，且 start 相同 ⇒ 同一"年初"）
            prev_end = next((e for e in ends
                             if 80 <= item["days"] - ends[e][0]["days"] <= 100
                             and e < end), None)
            if prev_end is not None:
                # (a) 口径一致性硬约束：prev 的申报日不得晚于 item 的申报日
                cands = [v for v in ends[prev_end] if v["filed"] <= item["filed"]]
                if cands:
                    prev = _pick(cands, vintage)
                    val = item["val"] - prev["val"]
                    basis = f"ytd_diff({prev['bucket']}→{item['bucket']})"
            elif item["bucket"] == "q":
                val = item["val"]                       # 财年第一季：3M 本身就是单季
                basis = "direct_quarter"

            if val is None or not _is_quarter_end(end):
                continue
            q = _qlabel(end)
            chain_q[end] = (q, val)
            filed = max([item["filed"]] + ([prev["filed"]] if prev else []))
            rec = {
                "quarter": q, "period_end": end, "value": val, "tag": item["tag"],
                "basis": basis, "first_filed": filed,
                "latest_filed": per_end_latest.get(end, filed),
                "forms": "+".join(sorted({item["form"]} | ({prev["form"]} if prev else set()))),
                "n_versions": len(ends[end]) + (len(ends[prev_end]) if prev else 0),
                "prev_tag": prev["tag"] if prev else item["tag"],
            }
            old = out.get(q)
            if old is None or filed >= old["first_filed"]:
                if old is not None and old["period_end"] != end:
                    print(f"    ! {tag} {q} 同一季度两个期末（{old['period_end']}/{end}），保留后者")
                out[q] = rec

        # 内部一致性校验：一条覆盖完整财年的链上，四个单季之和必须等于 12M 的 FY fact。
        # 这是对 YTD 差分最直接的验证，不依赖任何外部常识数字。
        fy_items = [(_pick(ends[e], vintage)) for e in sorted(ends)
                    if _pick(ends[e], vintage)["bucket"] == "fy"]
        if fy_items and set(chain_q) >= set(list(sorted(ends))[-4:]):
            fy = fy_items[-1]
            parts = [chain_q[e][1] for e in sorted(chain_q) if e <= fy["end"]][-4:]
            if len(parts) == 4 and abs(sum(parts) - fy["val"]) / fy["val"] > 0.005:
                checks.append({"tag": tag, "fy_end": fy["end"], "fy_val": fy["val"],
                               "q_sum": sum(parts),
                               "rel": (sum(parts) - fy["val"]) / fy["val"]})
    return out, checks


def _merge_tags(series: dict[str, dict[str, dict]], names: list[str], label: str,
                vintage: str, order: list[str] | None = None) -> dict[str, dict]:
    """跨 tag（等价组）逐季度择值。

    as_reported：取 first_filed 最早者（真正的 point-in-time，避免重述口径混入历史序列）；
    latest：按 tag 优先级取（重述后口径）。
    优先级：一个组的名次 = 组内最高优先级 tag 的名次（order 为 tag 优先级列表）。
    """
    order = order or names

    def rank(name: str) -> int:
        return min(order.index(t) for t in name.split("+") if t in order)

    quarters = sorted({q for s in series.values() for q in s})
    merged: dict[str, dict] = {}
    for q in quarters:
        cands = [(n, series[n][q]) for n in names if q in series[n]]
        if not cands:
            continue
        if vintage == "as_reported":
            earliest = min(c["first_filed"] for _, c in cands)
            near = [c for c in cands
                    if (date.fromisoformat(c[1]["first_filed"])
                        - date.fromisoformat(earliest)).days <= TIE_DAYS]
            chosen = min(near, key=lambda c: rank(c[0]))[1]
        else:
            chosen = min(cands, key=lambda c: rank(c[0]))[1]
        merged[q] = chosen
    return merged


# --------------------------------------------------------- 单公司流程 --------
def _equivalence_groups(series: dict[str, dict[str, dict]], tags: list[str],
                        raw: dict[str, list[dict]], tol: float = 0.005) -> list[list[str]]:
    """把 tag 按"是否可证明同口径"分组（并查集）。

    判据：两个 tag 在**重叠期**上的取值差异均 <= tol。优先用单季序列比较；
    无单季重叠时退化到原始区间 fact（同一 start/end 区间）比较。
    同一组的 tag 才允许在 YTD 链条里互相拼接（例如 GOOGL 的 Revenues 与
    RevenueFromContractWithCustomerExcludingAssessedTax 在重叠期完全一致 → 可拼接，
    从而把 2020Q4/2021Q4 的 first_filed 从 2023-02-03 提前到真实的 2021-02-02）；
    不可证明等价的（如 AMZN 两个 capex tag 在 FY2016 差 15.8%）则严格不混算。
    """
    def fact_vals(t: str) -> dict[tuple[str, str], float]:
        out = {}
        for u in raw[t]:
            form = str(u.get("form") or "")
            if not u.get("start") or not FORM_RE.match(form):
                continue
            days = (date.fromisoformat(u["end"]) - date.fromisoformat(u["start"])).days
            if _bucket(days):
                out.setdefault((u["start"], u["end"]), float(u["val"]))
        return out

    fv = {t: fact_vals(t) for t in tags}
    parent = {t: t for t in tags}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            ov = sorted(set(series[a]) & set(series[b]))
            if ov:
                pairs = [(a, b, q, series[a][q]["value"], series[b][q]["value"]) for q in ov]
                scale = "单季"
            else:
                keys = sorted(set(fv[a]) & set(fv[b]))
                pairs = [(a, b, f"{k[0]}~{k[1]}", fv[a][k], fv[b][k]) for k in keys]
                scale = "区间"
            if not pairs:
                print(f"      [等价性] {a[:36]} vs {b[:36]}: 无任何重叠数据，不拼接（无法证明等价）")
                continue
            worst = max(abs(x - y) / (abs(x) or 1.0) for _, _, _, x, y in pairs)
            if worst <= tol:
                parent[find(a)] = find(b)
                print(f"      [等价性] {a[:36]} ≡ {b[:36]}：重叠 {len(pairs)} 个{scale}，"
                      f"最大差异 {worst*100:.2f}% → 可跨 tag 拼接")
            else:
                wq = max(pairs, key=lambda p: abs(p[3] - p[4]) / (abs(p[3]) or 1.0))
                print(f"      [等价性] {a[:36]} ≢ {b[:36]}：重叠 {len(pairs)} 个{scale}，"
                      f"最大差异 {worst*100:.2f}%（如 {wq[2]}: {wq[3]/1e9:,.3f}B vs {wq[4]/1e9:,.3f}B）"
                      f" → **不拼接**（口径不同，避免混算）")
    groups: dict[str, list[str]] = {}
    for t in tags:
        groups.setdefault(find(t), []).append(t)
    return [sorted(g, key=lambda t: tags.index(t)) for g in groups.values()]


def collect_company(ticker: str, cik: str) -> tuple[pd.DataFrame, dict]:
    print(f"\n-- {ticker} (CIK {cik}) --")
    picked: dict[str, dict[str, dict]] = {}
    alt: dict[str, dict[str, dict]] = {}       # 另一种口径（仅用于差异诊断）
    summary: dict[str, dict] = {}
    other = "latest" if VINTAGE == "as_reported" else "as_reported"
    for metric, tags in METRIC_TAGS.items():
        raw: dict[str, list[dict]] = {}
        series_by_tag: dict[str, dict[str, dict]] = {}
        for tag in tags:
            facts = _usd_facts(_fetch_tag(ticker, cik, tag))
            if not facts:
                continue
            for f in facts:
                f["_tag"] = tag
            raw[tag] = facts
            series_by_tag[tag] = _quarterly(facts, tag, VINTAGE)[0]
        live = [t for t in tags if t in raw]
        if not live:
            picked[metric], alt[metric] = {}, {}
            print(f"   {metric:8s}: 无数据")
            continue

        print(f"   [{metric}] tag 等价性检查:")
        groups = _equivalence_groups(series_by_tag, live, raw)

        # 每个等价组作为一条候选序列（组内可跨 tag 拼接 YTD 链条）
        series: dict[str, dict[str, dict]] = {}
        series_alt: dict[str, dict[str, dict]] = {}
        all_checks: list[dict] = []
        for gi, grp in enumerate(groups):
            facts = [f for t in grp for f in raw[t]]
            name = "+".join(grp)
            series[name] = _quarterly(facts, name, VINTAGE)[0]
            all_checks += _quarterly(facts, name, VINTAGE)[1]
            series_alt[name] = _quarterly(facts, name, other)[0]
        group_names = list(series)

        recs = _merge_tags(series, group_names, f"{ticker}/{metric}", VINTAGE, order=live)
        alt[metric] = _merge_tags(series_alt, group_names, f"{ticker}/{metric}[{other}]", other, order=live)

        for q, r in recs.items():                       # capex 符号统一为"正数=支出"
            if r["value"] < 0:
                print(f"    ! {ticker} {q} {metric} 为负值 {r['value']:,.0f}，取绝对值")
                r["value"] = abs(r["value"])
        picked[metric] = recs
        used = sorted({r["tag"] for r in recs.values()})
        qs = sorted(recs)
        print(f"   {metric:8s}: {len(recs)} 个季度 {qs[0]}→{qs[-1]}；使用 tag: {', '.join(used)}")
        n_bad = len(all_checks)
        print(f"      YTD 差分内部校验（四单季之和 = 10-K 全年 fact）："
              f"{'全部通过' if n_bad == 0 else f'{n_bad} 处不符'}")
        for c in all_checks:
            print(f"        !! {c['tag']} FY{c['fy_end']}: 单季合计 {c['q_sum']/1e9:,.3f}B "
                  f"vs 10-K 全年 {c['fy_val']/1e9:,.3f}B（{c['rel']*100:+.2f}%）")
        # 跨 tag 拼接的季度（链条内 item 与 prev 来自不同 tag）
        cross = [(q, r["prev_tag"], r["tag"]) for q, r in recs.items() if r["prev_tag"] != r["tag"]]
        if cross:
            print(f"      [跨 tag 拼接] {len(cross)} 期（组内已证明等价）："
                  + ", ".join(f"{q}:{a.split('With')[-1][:18]}→{b.split('With')[-1][:18]}" for q, a, b in sorted(cross)[:6])
                  + (" …" if len(cross) > 6 else ""))
        summary[metric] = {"tags": used, "n": len(recs), "range": [qs[0], qs[-1]],
                           "bases": sorted({r["basis"] for r in recs.values()}),
                           "n_check_fail": n_bad, "n_cross_tag": len(cross),
                           "groups": groups}
        # tag 切换点（定义可能变化的年份，必须人工确认）
        seq = [(q, recs[q]["tag"]) for q in qs]
        switches = [(seq[i - 1][0], seq[i][0], seq[i - 1][1], seq[i][1])
                    for i in range(1, len(seq)) if seq[i][1] != seq[i - 1][1]]
        for a, b, t0, t1 in switches:
            print(f"      [tag 切换] {a}({t0.split('PaymentsToAcquire')[-1][:24]}) → "
                  f"{b}({t1.split('PaymentsToAcquire')[-1][:24]})：注意口径可能变化")
        summary[metric]["switches"] = switches

    # 重述差异诊断：选定口径 vs 对照口径
    if alt:
        print(f"   [诊断] {VINTAGE} vs {other} 口径差异（>0.5%）:")
        n = 0
        for metric in METRIC_TAGS:
            for q in sorted(set(picked[metric]) & set(alt.get(metric, {}))):
                a, b = picked[metric][q]["value"], alt[metric][q]["value"]
                if a and abs(b - a) / abs(a) > 0.005:
                    print(f"      {metric} {q}: {VINTAGE}={a/1e9:,.3f}B → {other}={b/1e9:,.3f}B "
                          f"({(b/a-1)*100:+.1f}%)")
                    n += 1
        if n == 0:
            print("      无（各期两种口径一致）")

    quarters = sorted(set(picked["revenue"]) | set(picked["capex"]))
    rows = []
    for q in quarters:
        rv, cx = picked["revenue"].get(q), picked["capex"].get(q)
        if not rv and not cx:
            continue
        end = (rv or cx)["period_end"]
        if rv and cx and rv["period_end"] != cx["period_end"]:
            print(f"    ! {ticker} {q} 收入/资本开支期末不一致：{rv['period_end']} vs {cx['period_end']}")
        firsts = [r["first_filed"] for r in (rv, cx) if r]
        latest = [r["latest_filed"] for r in (rv, cx) if r]
        rows.append({
            "quarter": q, "period_end": end, "company": ticker,
            "revenue": rv["value"] if rv else None,
            "capex": cx["value"] if cx else None,
            "tag_revenue": rv["tag"] if rv else None,
            "tag_capex": cx["tag"] if cx else None,
            "first_filed": max(firsts) if firsts else None,
            "latest_filed": max(latest) if latest else None,
            "source_url": CONCEPT_URL.format(cik=cik, tag=(rv or cx)["tag"]),
        })
    return pd.DataFrame(rows), summary


# ------------------------------------------------------------- 同比 --------
def add_yoy(df: pd.DataFrame, cols: list[str], key: str | None = None) -> pd.DataFrame:
    """同比按"去年同一日历季度"对齐（非 shift(4)），缺季度时不会错位。"""
    df = df.copy()
    y = df["quarter"].str[:4].astype(int)
    df["_prev_q"] = (y - 1).astype(str) + df["quarter"].str[4:]
    if key is None:
        right = df.set_index("quarter")
        for c in cols:
            df[f"{c}_yoy_pct"] = (df[c] / df["_prev_q"].map(right[c]) - 1) * 100
    else:
        right = df.set_index([key, "quarter"])
        for c in cols:
            prev = [right[c].get((k, p)) for k, p in zip(df[key], df["_prev_q"])]
            df[f"{c}_yoy_pct"] = [
                (a / b - 1) * 100 if (a == a and b == b and b) else None
                for a, b in zip(df[c], prev)
            ]
    return df.drop(columns=["_prev_q"])


def build_panel(quarterly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q, g in quarterly.groupby("quarter", sort=True):
        rev, cap = g["revenue"].dropna(), g["capex"].dropna()
        firsts = g["first_filed"].dropna().tolist()
        rows.append({
            "quarter": q, "period_end": g["period_end"].max(),
            "n_companies": int((g["revenue"].notna() | g["capex"].notna()).sum()),
            "n_revenue": int(len(rev)), "n_capex": int(len(cap)),
            "revenue_sum": float(rev.sum()) if len(rev) else None,
            "capex_sum": float(cap.sum()) if len(cap) else None,
            "first_filed": max(firsts) if firsts else None,
        })
    panel = pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)
    # 合计同比：仅当去年同期覆盖的公司数一致时才可比（否则口径漂移会造出假同比）
    right = panel.set_index("quarter")
    pqs = [f"{int(q[:4]) - 1}{q[4:]}" for q in panel["quarter"]]
    for col in ["revenue_sum", "capex_sum"]:
        n_col = "n_revenue" if col == "revenue_sum" else "n_capex"
        vals = []
        for row, pq in zip(panel.to_dict("records"), pqs):
            ok = (pq in right.index and right.loc[pq, n_col] == row[n_col]
                  and row[col] and right.loc[pq, col])
            vals.append((row[col] / right.loc[pq, col] - 1) * 100 if ok else None)
        panel[col.replace("_sum", "_yoy_pct")] = vals
    panel = panel.drop(columns=["n_revenue", "n_capex"])
    panel["legal_basis"] = LEGAL_BASIS["sec"]
    return panel[["quarter", "period_end", "n_companies", "revenue_sum", "capex_sum",
                  "revenue_yoy_pct", "capex_yoy_pct", "first_filed", "legal_basis"]]


# -------------------------------------------------------- 质量校验 --------
# 公开常识锚点（公司 10-K 数字），用于验证 YTD 差分没算错。
# 口径说明（实测得出，见报告）：
#   * MSFT / GOOGL 的 tag 就是现金流量表"购置固定资产 (Additions/Purchases of PP&E)"行，
#     不含融资租赁 → 与 10-K 该行数字逐年吻合。
#   * AMZN 的 PaymentsToAcquireProductiveAssets 在 FY2017–2022 与公开引用的
#     "Purchases of property and equipment"（11.955 / 13.427 / 16.861 / 40.140 / 61.053 / 63.645B）
#     逐年完全一致；但 FY2023 起该 tag 值 = 现金口径 + 当年融资租赁新增
#     （48.133+4.596=52.729B，77.658+5.341=82.999B），即口径在 FY2023 发生了变化。
#     故 AMZN 锚点用 tag 自身口径（82.999B，与"含融资租赁约 830 亿美元"一致），
#     同时打印现金口径差异，避免读者误判。
ANCHORS = [
    ("MSFT FY2024 (2023-07~2024-06) capex 四季合计 / 10-K $44,477M", "MSFT", "capex", "2023-07-01", "2024-06-30", 44.477e9, 0.03, 4),
    ("MSFT FY2024 收入四季合计 / 10-K $245,122M", "MSFT", "revenue", "2023-07-01", "2024-06-30", 245.122e9, 0.01, 4),
    ("MSFT FY2025 (2024-07~2025-06) capex 四季合计（公开约 $64.6B，不含融资租赁）", "MSFT", "capex", "2024-07-01", "2025-06-30", 64.6e9, 0.10, 4),
    ("AMZN 2024 全年 capex(tag 口径=现金 PP&E+融资租赁) $82,999M", "AMZN", "capex", "2024-01-01", "2024-12-31", 82.999e9, 0.02, 4),
    ("AMZN 2023 全年 capex(同口径) $52,729M", "AMZN", "capex", "2023-01-01", "2023-12-31", 52.729e9, 0.02, 4),
    ("AMZN 2022 全年 capex(同口径=公开现金口径) $63,645M", "AMZN", "capex", "2022-01-01", "2022-12-31", 63.645e9, 0.02, 4),
    ("AMZN 2024 全年收入 / 10-K $637,959M", "AMZN", "revenue", "2024-01-01", "2024-12-31", 637.959e9, 0.01, 4),
    ("GOOGL 2024 全年 capex(PP&E) / 10-K $52,535M", "GOOGL", "capex", "2024-01-01", "2024-12-31", 52.535e9, 0.03, 4),
    ("GOOGL 2024 全年收入 / 10-K $350,018M", "GOOGL", "revenue", "2024-01-01", "2024-12-31", 350.018e9, 0.01, 4),
]


def cross_check(df: pd.DataFrame) -> None:
    print("\n== 交叉核对（对公开财报已知数字）==")
    for label, comp, col, lo, hi, exp, tol, n_expect in ANCHORS:
        sub = df[(df["company"] == comp) & (df["period_end"] >= lo) & (df["period_end"] <= hi)]
        got = sub[col].dropna()
        if len(got) != n_expect:
            print(f"  ? {label}: 只凑齐 {len(got)}/{n_expect} 个季度，无法核对")
            continue
        s = float(got.sum())
        dev = (s - exp) / exp * 100
        print(f"  {'OK ' if abs(dev) <= tol * 100 else '!! '}{label}: 实测 {s/1e9:,.2f}B，差 {dev:+.2f}%")

    # AMZN capex 口径敏感性：公开引用的"现金 capex"（10-K 现金流量表行）对比
    print("  [AMZN capex 口径] 本列 = tag 口径；公开现金口径（不含融资租赁）:")
    cash = {"2022": 63.645, "2023": 48.133, "2024": 77.658}
    for y, v in cash.items():
        sub = df[(df["company"] == "AMZN") & (df["period_end"] >= f"{y}-01-01")
                 & (df["period_end"] <= f"{y}-12-31")]["capex"].dropna()
        if len(sub) == 4:
            got_ = float(sub.sum()) / 1e9
            print(f"      {y}: 本脚本 {got_:,.3f}B vs 现金口径 {v:,.3f}B"
                  f"（差 {(got_ - v):+,.3f}B = 当年融资租赁新增）")


def coverage_report(df: pd.DataFrame) -> None:
    print("\n== 季度覆盖与缺口 ==")
    def qidx(q: str) -> int:
        return int(q[:4]) * 4 + int(q[5:]) - 1
    for comp, g in df.groupby("company"):
        g = g.sort_values("quarter")
        qs = g["quarter"].tolist()
        have = {qidx(q) for q in qs}
        miss = [f"{i // 4}Q{i % 4 + 1}" for i in sorted(set(range(qidx(qs[0]), qidx(qs[-1]) + 1)) - have)]
        print(f"  {comp}: {qs[0]} → {qs[-1]}，共 {len(qs)} 个季度；区间内缺口 {len(miss)} 个：{miss or '无'}")
        miss_r = g[g["revenue"].isna()]["quarter"].tolist()
        miss_c = g[g["capex"].isna()]["quarter"].tolist()
        if miss_r:
            print(f"     收入缺失: {miss_r}")
        if miss_c:
            print(f"     资本开支缺失: {miss_c}")
        bad = g[g["first_filed"].notna() & (g["first_filed"] < g["period_end"])]
        print(f"     无前视校验（first_filed >= period_end）：{'通过' if len(bad) == 0 else f'失败 {len(bad)} 行'}")
        if len(bad):
            print(bad[["quarter", "period_end", "first_filed"]].to_string(index=False))


# ------------------------------------------------------------- 主流程 --------
def main() -> None:
    print(f"== 超大规模云厂商景气因子采集开始（SEC EDGAR XBRL companyconcept，口径={VINTAGE}）==")
    frames = []
    for ticker, cik, _ in COMPANIES:
        df, _ = collect_company(ticker, cik)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True).sort_values(["company", "period_end"]).reset_index(drop=True)
    df = add_yoy(df, ["revenue", "capex"], key="company")
    df["legal_basis"] = LEGAL_BASIS["sec"]
    df = df[["quarter", "period_end", "company", "revenue", "capex",
             "revenue_yoy_pct", "capex_yoy_pct", "tag_revenue", "tag_capex",
             "first_filed", "latest_filed", "source_url", "legal_basis"]]

    panel = build_panel(df)
    coverage_report(df)
    cross_check(df)

    print("\n== 写出 ==")
    for p in write_table(df, "cloud_complex_quarterly"):
        print("  ", p, f"({len(df)} 行)")
    for p in write_table(panel, "cloud_complex_panel"):
        print("  ", p, f"({len(panel)} 行)")

    print("\n== 最新 4 个季度 ==")
    for comp, _, _ in COMPANIES:
        sub = df[df["company"] == comp].tail(4)
        print(f"\n{comp}:")
        print(sub[["quarter", "revenue", "revenue_yoy_pct", "capex", "capex_yoy_pct",
                   "tag_revenue", "tag_capex", "first_filed", "latest_filed"]].to_string(index=False))
    print("\n合计面板（最新 6 个季度）：")
    print(panel.tail(6)[["quarter", "n_companies", "revenue_sum", "revenue_yoy_pct",
                         "capex_sum", "capex_yoy_pct", "first_filed"]].to_string(index=False))
    print(f"\n完成：季度表 {len(df)} 行（{df['quarter'].min()} → {df['quarter'].max()}）；"
          f"面板 {len(panel)} 行（{panel['quarter'].min()} → {panel['quarter'].max()}）")


if __name__ == "__main__":
    main()

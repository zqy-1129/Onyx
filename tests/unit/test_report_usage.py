"""`onyx report usage` 的聚合与渲染（S38）。

这一步的核心不是"能不能出表"，而是**CLI 与看板必须算出同一个数**。
所以这里既测数字，也测两件容易悄悄分叉的事：p50 用的是唯一那份分位数定义，
阈值用的是 `reconciler` 那一个常量。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from onyx.llm.measurement.reconciler import DEFAULT_DRIFT_MIN_TOKENS, DEFAULT_DRIFT_THRESHOLD
from onyx.llm.measurement.stats import median
from onyx.report.usage_report import (
    CSV_COLUMNS,
    RATE_COLUMNS,
    UsageOverview,
    _csv_cell,
    _md_cell,
    _num,
    build_overview,
    normalize_since,
    render_csv,
    render_markdown,
    render_table,
)
from onyx.store.db import Database
from onyx.store.records import TokenPartRecord, TraceRecord, UsageRecord
from onyx.store.repos import TraceRepo, UsageRepo


@pytest.fixture
def db(tmp_path):
    with Database(tmp_path / "usage.sqlite") as database:
        yield database


def _seed(db, *, trace_id: str, started_at: str, in_tokens: int, out_tokens: int,
          drift_pct: float | None = None, source: str = "engine",
          confidence: str = "high", model_id: str = "ollama-local/qwen3.5:9b") -> None:
    TraceRepo(db).upsert(TraceRecord(
        id=trace_id, kind="generation", purpose="chat", started_at=started_at,
        status="ok", provider_id="ollama-local", model_id=model_id, model_name="qwen3.5:9b",
    ))
    UsageRepo(db).upsert(UsageRecord(
        trace_id=trace_id, source=source, confidence=confidence,
        in_tokens=in_tokens, out_tokens=out_tokens, drift_pct=drift_pct,
    ))


def test_overview_totals(db):
    _seed(db, trace_id="t1", started_at="2026-10-05T10:00:00+00:00", in_tokens=100, out_tokens=20)
    _seed(db, trace_id="t2", started_at="2026-10-05T11:00:00+00:00", in_tokens=50, out_tokens=5)
    overview = build_overview(UsageRepo(db))
    assert (overview.traces, overview.in_tokens, overview.out_tokens) == (2, 150, 25)
    assert overview.by_source == {"engine": 2}
    assert overview.drift["threshold"] == DEFAULT_DRIFT_THRESHOLD, "阈值要能自解释"


def test_drift_p50_uses_the_single_percentile_definition(db):
    """曾经 API 用 `ordered[n//2]`：偶数个样本时它给出的是"上中位数"，与 stats.median 不同。

    这四个值特意选成两种算法结果不同（0.05 vs 0.06）——否则这条测试什么都没测。
    """
    for i, drift in enumerate((0.02, 0.04, 0.06, 0.40)):
        _seed(db, trace_id=f"t{i}", started_at=f"2026-10-05T{i:02}:00:00+00:00",
              in_tokens=10, out_tokens=1, drift_pct=drift)
    drift = build_overview(UsageRepo(db)).drift
    assert drift["p50"] == pytest.approx(median([0.02, 0.04, 0.06, 0.40]))
    assert drift["p50"] == pytest.approx(0.05)
    assert drift["p50"] != 0.06, "又用回 ordered[n//2] 了"
    assert drift["n"] == 4 and drift["max"] == pytest.approx(0.40)


def test_over_threshold_is_strict_against_the_shared_constant(db):
    """恰好等于阈值不算超（与 reconciler 报 TOKEN_DRIFT 的判据同一条边界）。"""
    _seed(db, trace_id="a", started_at="2026-10-05T01:00:00+00:00", in_tokens=10, out_tokens=1,
          drift_pct=DEFAULT_DRIFT_THRESHOLD)
    _seed(db, trace_id="b", started_at="2026-10-05T02:00:00+00:00", in_tokens=10, out_tokens=1,
          drift_pct=DEFAULT_DRIFT_THRESHOLD + 0.001)
    drift = build_overview(UsageRepo(db)).drift
    assert drift["over_threshold"] == 1


def test_the_summary_says_which_rule_it_counted_with(db):
    """`over_threshold` 与 `TOKEN_DRIFT` 异常数不是同一个量（后者还要求绝对差 ≥ min_tokens）。

    真机上 2298 个样本：相对差超阈 721 条、异常 711 条。**只差 10 条也是两个量**——
    被绝对差门槛挡掉的那几条是 in=120、pct=0.15（绝对差 16~18 tok），
    短 prompt 上两个门槛会分得很开，所以口径必须跟数一起出。
    """
    _seed(db, trace_id="a", started_at="2026-10-05T01:00:00+00:00", in_tokens=10, out_tokens=1,
          drift_pct=0.5)
    drift = build_overview(UsageRepo(db)).drift
    assert drift["min_tokens_for_anomaly"] == DEFAULT_DRIFT_MIN_TOKENS
    assert "TOKEN_DRIFT" in drift["rule"] and "绝对差" in drift["rule"]
    assert "pct 超阈" in render_markdown(build_overview(UsageRepo(db)))


def test_filters_apply(db):
    _seed(db, trace_id="old", started_at="2026-09-01T00:00:00+00:00", in_tokens=99, out_tokens=9)
    _seed(db, trace_id="new", started_at="2026-10-05T00:00:00+00:00", in_tokens=10, out_tokens=1,
          model_id="ollama-local/other")
    assert build_overview(UsageRepo(db), since="2026-10-01T00:00:00+00:00").traces == 1
    only = build_overview(UsageRepo(db), model="ollama-local/other")
    assert only.traces == 1 and only.model == "ollama-local/other"


def test_empty_range_says_no_traces_instead_of_showing_zero_cost(db):
    """"这段时间没跑"与"跑了一次但没花 token"是两件事，空表不许被画成一条 0 的线。"""
    overview = build_overview(UsageRepo(db))
    assert overview.traces == 0 and overview.timeseries == []
    assert any("没有 trace" in line for line in render_table(overview))
    assert "（空）" in render_markdown(overview)


def test_csv_header_is_pinned_and_commas_are_escaped():
    """列序漂了下游脚本会静默读错列，而 diff 里只看得出"换了个顺序"。"""
    overview = UsageOverview(since=None, model=None, bucket_minutes=60, traces=0,
                             in_tokens=0, out_tokens=0, thinking_tokens=0,
                             timeseries=[{"bucket": "2026-10-05T10:00", "traces": 3,
                                          "in_tokens": 1200, "out_tokens": 45,
                                          "decode_tps": 30.5, "cold_prefill_tps": None,
                                          "warm_prefill_tps": 900.25}])
    lines = render_csv(overview).strip().split("\n")
    assert lines[0] == ",".join(CSV_COLUMNS)
    assert lines[1].startswith("2026-10-05T10:00,3,1200,45,30.5000,,900.2500")

    awkward = UsageOverview(since=None, model=None, bucket_minutes=60, traces=0,
                            in_tokens=0, out_tokens=0, thinking_tokens=0,
                            timeseries=[{"bucket": "a,b", "traces": 1, "in_tokens": 0,
                                         "out_tokens": 0, "decode_tps": 0,
                                         "cold_prefill_tps": 0, "warm_prefill_tps": 0}])
    assert render_csv(awkward).split("\n")[1].startswith('"a,b"'), "含逗号的值必须转义"


def test_the_api_returns_exactly_the_same_numbers(db):
    """CLI 与看板的同源不是"看起来一样"，是同一个函数——这条断言把它钉住。"""
    from onyx.api.routes.traces import usage_summary

    _seed(db, trace_id="t1", started_at="2026-10-05T10:00:00+00:00", in_tokens=100,
          out_tokens=20, drift_pct=0.5)
    state = SimpleNamespace(usage=UsageRepo(db))
    view = usage_summary(since=None, model=None, bucket_minutes=60, state=state)
    assert view.model_dump() == build_overview(UsageRepo(db), bucket_minutes=60).as_dict()


def test_part_records_are_not_double_counted(db):
    """parts 与 usage 是两张表；汇总只数 usage，分段留给 `token explain`。

    写这条是因为"把分段加进 in_tokens"看起来像修了一个 bug，实际造出第二个事实源。
    """
    _seed(db, trace_id="t1", started_at="2026-10-05T10:00:00+00:00", in_tokens=100, out_tokens=20)
    UsageRepo(db).replace_parts("t1", [TokenPartRecord(trace_id="t1", part="system", tokens=1000)])
    overview = build_overview(UsageRepo(db))
    assert overview.in_tokens == 100


# ── --since 的校验（真机踩出来的：`--since 7d` 静默给出一张空表）──────
def test_since_is_normalized_to_the_shape_the_db_actually_compares(db):
    """库里 `started_at` 只有一种形状（定长 UTC + 微秒 + `+00:00`），过滤是字典序比较。"""
    _seed(db, trace_id="t1", started_at="2026-10-05T10:00:00+00:00", in_tokens=100, out_tokens=20)
    assert build_overview(UsageRepo(db), since="2026-10-01").since == "2026-10-01T00:00:00+00:00"
    assert build_overview(UsageRepo(db), since="2026-10-01").traces == 1
    # 不带时区的日期时间按 UTC 理解（写的人与库里存的是同一个意思）
    assert normalize_since("2026-10-01T08:30:00") == "2026-10-01T08:30:00+00:00"
    # 带别国时区的写法必须先折 UTC：`+08:00` 的串与 `+00:00` 的串比大小不是同一回事
    assert normalize_since("2026-10-01T00:00:00+08:00") == "2026-09-30T16:00:00+00:00"
    assert normalize_since("2026-09-29T06:00:00.000Z") == "2026-09-29T06:00:00+00:00"
    assert normalize_since(None) is None and normalize_since("   ") is None


def test_the_rate_columns_are_pinned_and_actually_exist(db):
    """`RATE_COLUMNS` 里写错一个字母，规则就**静默失效**（列名对不上 ⇒ 永远不匹配）。

    所以这里不测行为只测形状：三列都还在 `CSV_COLUMNS` 里，且非速率列不在其中。
    """
    assert set(RATE_COLUMNS) == {"decode_tps", "cold_prefill_tps", "warm_prefill_tps"}
    assert set(RATE_COLUMNS) <= set(CSV_COLUMNS), "有列名拼错了——那条规则现在什么都没守"
    assert set(CSV_COLUMNS) - RATE_COLUMNS == {"bucket", "traces", "in_tokens", "out_tokens"}


def test_a_populated_report_renders_rows_in_every_shape(db):
    """三种渲染的**数据行**都要真跑一遍：只有空表被测过的话，
    行模板里一个格式化写错（`{:>10,}` 写成 `{:>10}`）会在人读的输出里静默错位。"""
    _seed(db, trace_id="t1", started_at="2026-10-05T10:15:00+00:00", in_tokens=1200, out_tokens=45,
          drift_pct=0.5)
    _seed(db, trace_id="t2", started_at="2026-10-05T10:45:00+00:00", in_tokens=80, out_tokens=3)
    overview = build_overview(UsageRepo(db), bucket_minutes=60)
    assert len(overview.timeseries) == 1, "同一小时的两条要落进同一个桶"

    table = render_table(overview)
    rows = [line for line in table if line.startswith("2026-10-05T10")]
    assert len(rows) == 1, rows
    assert "1,280" in rows[0], "桶内要加起来，且千分位分隔还在"
    assert "没有 trace" not in "".join(table)
    assert "pct 超阈" in "\n".join(table)

    md = render_markdown(overview)
    body = [line for line in md.splitlines() if line.startswith("| 2026-")]
    assert len(body) == 1, md
    assert "—" in body[0], "没有延迟数据 ⇒ 那些列是「—」而不是 0"
    assert "（空）" not in md
    assert "trace 数：**2**" in md

    # 「—」的两个来源：None 与 0.0 都必须显示成「—」（0 t/s 不是测量结果，是没测到）；
    # 但这条规则**只许作用于速率列**——token 数真的是 0（空输出）时必须照写 0，
    # 否则"这一发没花 token"会被渲染成"没测"，那是把已知说成未知。
    assert _num(None) == _num(0.0) == _num(0) == "—" and _num(30.55) == "30.6"
    assert _md_cell("decode_tps", None) == _md_cell("cold_prefill_tps", 0.0) == "—"
    assert _md_cell("in_tokens", 0) == "0" and _md_cell("bucket", "engine") == "engine"
    assert _csv_cell("decode_tps", 0.0) == "" and _csv_cell("in_tokens", 0) == "0"

    # 真数值的浮点分支：库里那一发是 int（SQL 聚合），带小数的速率要留一位。
    # 取值刻意避开 .x5 的整半——`format` 走的是 round-half-even，测出来会像在测舍入规则。
    floats = UsageOverview(since=None, model=None, bucket_minutes=60, traces=1,
                           in_tokens=10, out_tokens=2, thinking_tokens=0,
                           timeseries=[{"bucket": "2026-10-05T10:00", "traces": 1,
                                        "in_tokens": 10, "out_tokens": 2,
                                        "decode_tps": 30.54, "cold_prefill_tps": None,
                                        "warm_prefill_tps": 900.26}])
    md_rows = [line for line in render_markdown(floats).splitlines() if line.startswith("| 2026-")]
    assert md_rows == ["| 2026-10-05T10:00 | 1 | 10 | 2 | 30.5 | — | 900.3 |"]


def test_a_beyond_midnight_offset_does_not_drop_an_in_range_trace(db):
    """`--since 2026-10-01T00:00:00+08:00` 的真实时刻是 09-30T16:00Z。

    不折算的话字典序会把它当成 10-01T00:00Z，**边界前的那一发被静默丢掉**——
    数字看起来完全合理，所以这条要把"未折算会怎样"一起断言出来。
    """
    _seed(db, trace_id="edge", started_at="2026-09-30T20:00:00+00:00", in_tokens=10, out_tokens=1)
    repo = UsageRepo(db)
    assert build_overview(repo, since="2026-10-01T00:00:00+08:00").traces == 1
    assert repo.summarize(since="2026-10-01T00:00:00+08:00").traces == 0, "对照：未折算就是丢数据"


@pytest.mark.parametrize("bad", ["7d", "24h", "昨天", "2026-13-45", "2026-02-30", "10/01/2026"])
def test_an_unparseable_since_raises_instead_of_showing_an_empty_report(db, bad):
    """形状对但日期不存在也得拦（`2026-13-45` 过得了正则，过不了 `date.fromisoformat`）。"""
    with pytest.raises(ValueError) as exc:
        build_overview(UsageRepo(db), since=bad)
    assert "--since" in str(exc.value) and bad in str(exc.value)


def test_the_empty_report_trap_is_real_not_decorative(db):
    """校验存在的理由必须能被复现：绕过校验直接查，`7d` 得到的是**一条都不剩**。

    如果把 `normalize_since` 里的校验删掉，`test_an_unparseable_since_raises...` 会红；
    这条则钉住"删掉校验之后世界错在哪"——空表 + `traces=0` 看起来像真的没有请求。
    """
    _seed(db, trace_id="t1", started_at="2026-10-05T10:00:00+00:00", in_tokens=100, out_tokens=20)
    assert UsageRepo(db).summarize(since="7d").traces == 0


def test_the_api_answers_400_for_a_bad_since(db):
    from fastapi import HTTPException

    from onyx.api.routes.traces import usage_summary

    _seed(db, trace_id="t1", started_at="2026-10-05T10:00:00+00:00", in_tokens=100, out_tokens=20)
    state = SimpleNamespace(usage=UsageRepo(db))
    with pytest.raises(HTTPException) as exc:
        usage_summary(since="7d", model=None, bucket_minutes=60, state=state)
    assert exc.value.status_code == 400 and "7d" in exc.value.detail
    assert usage_summary(since=None, model=None, bucket_minutes=60, state=state).traces == 1

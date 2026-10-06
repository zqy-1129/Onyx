"""S40：真浏览器那一档（`-m browser`）。

**这里只测"只有浏览器能看见的东西"**。取数同源归 `-m e2e`，纯函数映射归 vitest；
把三者分开不是为了目录好看，是因为它们的失效形状不同：
接口层能给出 null，渲染层照样能把它写成 0；vitest 能证明格式化函数正确，
证明不了它被调用在那一列上。

八条用例的来源都是本项目真发生过的缺陷（SSE 静默失效、空面板被读成"没人试过"、
`[object Object]` 顶掉整列、折线为空桶掉到 0、按钮文字竖排、贴边框裁字、
子页只能手敲 hash、rail 状态刷新就丢）。

进程编排（起后端 + 起 vite + 收端口）在 `browser_stack.py`。
"""

from __future__ import annotations

import json
import re

import httpx
import pytest

from tests.e2e.browser_stack import open_app, wait_loaded

pytestmark = [pytest.mark.browser]

#: 六个主页面。marker 挑的是**这一页独有的静态文案**（不是数据值——种的是 mock 模型，
#: 拿 "qwen" 当锚点会在任何测试栈上都红；导航条六页都在，等它只能证明应用起来了）
PAGES = [
    ("fleet", "已载入模型"),
    ("models", "模型（"),
    ("playground", "发送"),
    ("traces", "已加载"),
    ("usage", "采信来源分布"),
    ("tools", "工具注册表"),
]


def _main(page) -> str:
    return page.locator("main").inner_text()


def test_every_page_renders_data_rather_than_an_empty_panel(page, web_url):
    """空面板会被读成"没人试过"，而真相常常是"取到了但没渲染"（S25 那次）。

    所以这里不只断言"有字"，还断言**这一页自己的数出现了**：
    先从 API 拿到期望的数字，再要求它出现在渲染出来的文本里。
    """
    api = web_url.api
    ledger = httpx.get(f"{api}/api/usage/summary", timeout=10.0).json()
    traces = httpx.get(f"{api}/api/traces?limit=50", timeout=10.0).json()["items"]

    for route, marker in PAGES:
        open_app(page, web_url.base, route, wait_for=marker)
        wait_loaded(page)
        text = _main(page)
        assert page.locator(".skeleton").count() == 0, f"{route} 停在骨架屏：数据没回来也没报错"
        assert "加载失败" not in text and "Error" not in text, f"{route} 显示的是错误态"
        assert "[object Object]" not in text, f"{route} 又把对象直接 String() 了"
        # 不用字数当门槛（Playground 本来就只有一堆控件、178 字是正常形状）：
        # 要断言的是"这一页自己的东西在屏幕上"
        assert marker in text, f"{route} 等到的 marker 不在正文里（等错元素了）"

    # 数字同源：列表页要能数出 API 数出来的条数，Ledger 要印出自己那份汇总的 trace 数
    open_app(page, web_url.base, "traces", wait_for="已加载")
    # 这一页的"加载完了"不是骨架屏消失，而是计数真的写出来了——等错条件就会拿到"共 — 条 · 已加载 0"
    page.get_by_text(re.compile(r"共 \d+ 条")).first.wait_for(state="visible", timeout=20_000)
    assert str(len(traces)) in _main(page), "列表页的条数与 API 不一致（分页游标或筛选漂了）"
    open_app(page, web_url.base, "usage", wait_for="采信来源分布")
    wait_loaded(page)
    rendered = _main(page)
    assert f"{ledger['traces']:,}" in rendered or str(ledger["traces"]) in rendered, \
        "Ledger 顶部那个请求数没取自同一份汇总"


def test_ledger_never_draws_zero_for_a_rate_it_did_not_measure(page, web_url):
    """S39 那半条的浏览器版本：SQL 出 NULL 之后，渲染层照样能把它写成 0。

    断言完全由数据算出来——先问 API 哪一列全是 null，再要求那一行显示「—」。
    一列都没有全空时**这条要红**：那说明种子不再经过这条路径，测试就成了装饰。
    """
    summary = httpx.get(f"{web_url.api}/api/usage/summary", timeout=10.0).json()
    buckets = summary["timeseries"]
    assert len(buckets) >= 2, f"时序只有 {len(buckets)} 个桶，画不出折线——种子里的时间跨度失效了"

    empty = [(col, label) for col, label in (("warm_prefill_tps", "warm prefill"),
                                             ("cold_prefill_tps", "cold prefill"))
             if all(b.get(col) is None for b in buckets)]
    assert empty, "两列速率都有真数据了，这条守的就没东西可守——请改成断言 null 桶处断线"

    open_app(page, web_url.base, "usage", wait_for="采信来源分布")
    wait_loaded(page)
    text = _main(page)
    for column, label in empty:
        assert label in text, f"时序面板里没有 {column} 这一行（面板没渲染？）"
        assert "—" in text, f"{label} 全列没测到，页面上却没有「—」"
    assert "0.0 t/s" not in text and "0 t/s" not in text, "把「没测到」渲染成了一个看起来合理的 0"


def test_trace_detail_shows_the_equality_with_its_condition_and_tier(page, web_url, backend_url):
    """clamp 的那条 trace 必须同时给出：条件句、档位、残差、以及一条能跑的命令。"""
    trace_id = backend_url.clamped_trace_id
    detail = httpx.get(f"{web_url.api}/api/traces/{trace_id}", timeout=10.0).json()
    attribution = detail["attribution"]
    assert attribution.get("clamped") is True, "种出来的这条没走 clamp，种子失效了"

    open_app(page, web_url.base, f"traces/{trace_id}", wait_for="分段归因")
    wait_loaded(page)
    text = _main(page)
    assert "Σ分段 + template_ctl = 引擎计数" in text and "仅未 clamp 时成立" in text
    assert f"分段按 {attribution['count_source']} 数" in text, "档位必须与残差一起出现"
    assert "计数器高估" in text, "clamp 的那条不该只给一个笼统的未闭合"
    assert "onyx calibrate" in text and "mock/undercount" in text, \
        "给命令就要给抄得动的命令（带真模型名），占位符等于没给"
    assert "不回填" in text, "要当场说清历史行为什么还是这个样子"


def test_playground_request_shows_up_in_the_page_and_sse_is_live(page, web_url):
    """当年那个缺陷的正面覆盖：SSE 从上线起就静默失效，而所有测试都是绿的。

    两条独立证据：**● 已连接**（浏览器真的收下了那条流）与**结果出现在页面上**（这发真的走完了）。
    少了前一条，测试只能证明"发出去了"，证明不了"事件总线还活着"。
    """
    open_app(page, web_url.base, "playground", wait_for="发送")
    page.get_by_text("● 已连接").wait_for(state="visible", timeout=15_000)

    # 模型是靠 label 包着 checkbox 选的（`effective = selected`，不勾就没有可发的模型，
    # 而"发送"会一直 disabled——点文本节点不一定会触发，直接 check 那个 input）
    page.locator("label", has_text="mock/zh-转账").locator("input[type='checkbox']").check()
    page.locator("textarea.input").fill("北京现在天气怎么样？")
    page.get_by_role("button", name="发送").click()   # click 自己会等 enabled，写 wait_for(state=...) 是错的 API

    # 等到"这一发真的跑完了"的那个界面状态：结果存在时 RunPanel 才会给出「查看 trace →」。
    # 不用 `get_by_text("转账")` 当终点——模型名里就有这两个字，它一开始就可见，会把等待变成假通过。
    page.get_by_role("button", name="查看 trace →").wait_for(state="visible", timeout=30_000)
    text = _main(page)
    assert "trace" in text.lower(), "跑完了却连 trace id 都没有——这条链路在页面上不可核对"
    assert "engine" in text or "high" in text, "采信出处/置信度必须跟着数字上屏（R1）"
    assert page.locator(".code").first.inner_text().strip() == "转账", "输出区没有真的渲染这发的正文"


def test_sse_event_frames_reach_a_real_browser_subscriber(page, web_url, backend_url):
    """浏览器里的第二个订阅者必须收到**真事件帧**——这条是依赖总线的，摘掉 broker 就红。

    为什么不能只断言"Playground 把结果画出来了"：那个结果来自 POST 的返回值，
    事件总线死了它照样绿——而 S23 那次恰恰是"REST 全好、SSE 静默失效"。
    所以这里在页内自己开一个 `EventSource`，收**测试期间发出去的那一发请求**产生的帧。
    """
    open_app(page, web_url.base, "playground", wait_for="发送")
    # 应用自己的订阅先亮起来（连不上就是编排/代理断了，后面不必再看帧）
    page.get_by_text("● 已连接").wait_for(state="visible", timeout=15_000)

    page.evaluate(
        """() => { window.__frames = [];
                   const src = new EventSource('/api/stream');
                   src.onmessage = (e) => { try { window.__frames.push(JSON.parse(e.data)); }
                                            catch { /* 坏帧按契约丢弃 */ } };
                   window.__src = src; }"""
    )
    page.wait_for_function("() => window.__src && window.__src.readyState === 1", timeout=15_000)

    resp = httpx.post(f"{backend_url.base}/api/playground/chat",
                      json={"model": "mock/zh-转账", "prompt": "这条是给 SSE 的"}, timeout=30.0)
    assert resp.status_code == 200, resp.text
    trace_id = resp.json()["trace_id"]

    page.wait_for_function(
        f"() => window.__frames.some(f => f.trace_id === {json.dumps(trace_id)})",
        timeout=15_000)
    frames = page.evaluate("() => window.__frames.filter(f => f.trace_id).map(f => f.type)")
    assert any(kind in frames for kind in ("text_delta", "trace_end", "usage_engine")), \
        f"这一发的帧一条都没到浏览器：{frames[:8]}"
    page.evaluate("() => window.__src.close()")


def test_rail_open_state_survives_a_reload(page, web_url):
    """`onyx.rail-open` 是用户偏好：刷新就丢等于没有持久化（这条是明确要求过的）。"""
    open_app(page, web_url.base, "fleet", wait_for="已载入模型")
    wait_loaded(page)
    logo = page.locator(".rail-logo")
    before = logo.get_attribute("aria-expanded")
    logo.click()
    assert logo.get_attribute("aria-expanded") != before, "点开了却没改变展开态"
    after = logo.get_attribute("aria-expanded")
    assert page.evaluate("() => localStorage.getItem('onyx.rail-open')") is not None

    page.reload(wait_until="domcontentloaded")
    open_app(page, web_url.base, "fleet", wait_for="已载入模型")
    assert page.locator(".rail-logo").get_attribute("aria-expanded") == after, \
        "刷新之后开合状态回到了默认值——localStorage 没被读回来"


@pytest.mark.parametrize("width", [1280, 1600])
def test_no_horizontal_overflow_at_common_widths(page, web_url, width):
    """贴边框/裁字那类缺陷（9b97ae1）在 diff 里看不见，只能量出来。"""
    page.set_viewport_size({"width": width, "height": 900})
    for route, marker in PAGES:
        open_app(page, web_url.base, route, wait_for=marker)
        wait_loaded(page)
        metrics = page.evaluate(
            "() => ({sw: document.documentElement.scrollWidth,"
            " cw: document.documentElement.clientWidth})")
        assert metrics["sw"] <= metrics["cw"] + 1, \
            f"{route} 在 {width}px 下横向溢出：{json.dumps(metrics)}"


def test_labels_are_not_rendered_one_character_tall(page, web_url):
    """竖排（一个字一行）是同一类事故：单位标签 "t/s" 被挤成三行时，人只会觉得"排版怪"。

    这里量的是导航项与单位标签的宽高比——正常横排文字必然远宽于高。
    """
    open_app(page, web_url.base, "usage", wait_for="采信来源分布")
    wait_loaded(page)
    boxes = page.evaluate(
        """() => [...document.querySelectorAll('.rail-item, .nowrap, .legend-item')]
             .map(el => { const r = el.getBoundingClientRect();
                          return {t: (el.textContent || '').trim().slice(0, 12),
                                  w: Math.round(r.width), h: Math.round(r.height)}; })
             .filter(b => b.w > 0 && b.h > 0)"""
    )
    assert boxes, "一个可量的标签都没有——选择器漂了，这条会假绿"
    tall = [b for b in boxes if b["h"] > 0 and b["w"] / b["h"] < 1.0 and len(b["t"]) > 2]
    assert not tall, f"这些标签被压成了竖排：{tall[:5]}"

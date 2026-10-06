# Onyx 看板设计语言

对标 **Grafana / Datadog**：暗色优先、面板网格、高数据密度、等宽数字。
不是 LangSmith 那种 trace 中心的叙事式布局，也不是 Linear 的极简大留白——
因为排障时用户需要**同屏比较多个数字**，而不是逐个点开看。

---

## 1. 七条不可让步的规则

| # | 规则 | 理由 |
|---|---|---|
| R1 | **每个数字必须带出处徽标**（`source` + `confidence`） | 本项目与通用看板的根本差别。`engine/high` 与 `heuristic/low` 的 1842 是两个完全不同的东西 |
| R2 | **未知显示「—」，绝不显示 0** | 0 是一个测量值，「没测出来」不是。混淆二者会让人基于空数据做决策 |
| R3 | **冷/热永不合并**（表格分列、图分系列） | P11：热缓存 prefill 吞吐虚高 4.65×，合并后的 P50 既不代表冷启动也不代表稳态 |
| R4 | **颜色不是唯一通道**：色 + 符号 + 文字三重编码 | 色盲可及性；截图/打印/灰度终端下仍可读 |
| R5 | **红色只表示错误**，低置信度用灰 + 虚线下划线 | 把"不确定"画成红色会制造假告警，真告警反而被忽略 |
| R6 | **数字右对齐 + `tabular-nums`** | 列内可比；比例字体下 1842 与 8421 宽度不同，眼睛会对错行 |
| R7 | **异常必须可下钻到 trace，trace 必须可下钻到原始 body** | 没有证据链的数字无法被信任，也无法被反驳 |

---

## 2. 设计 token

```css
/* 背景：三层，越靠前越亮。层级差要小——大屏看板不该有强对比块 */
--bg-base:      #0B0E14;   /* 页面底 */
--bg-panel:     #11151D;   /* 面板 */
--bg-elevated:  #171C26;   /* 悬停/选中/下拉 */
--bg-inset:     #0E1219;   /* 代码块、原始 body */
--border:       #1F2733;
--border-strong:#2C3644;

/* 文字：三级足够，再多就没人分得清 */
--text-primary:   #E6E9EF;
--text-secondary: #9AA4B2;
--text-muted:     #626C7A;

/* 语义色（Grafana 系，饱和度压低以适应长时间观看） */
--ok:    #3FB950;   --ok-dim:    #1F3A28;
--warn:  #D29922;   --warn-dim:  #3A2F14;
--error: #F85149;   --error-dim: #3E1D1C;
--info:  #58A6FF;   --info-dim:  #16283D;
--accent:#A371F7;

/* 置信度：high=绿 medium=黄 low=灰+虚线（R5：不用红） */
/* prefill：cold=蓝 warm=橙（R3：两者必须同屏可辨） */

--font-ui:   Inter, system-ui, "PingFang SC", "Microsoft YaHei", sans-serif;
--font-mono: "JetBrains Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;

--space-1: 4px;  --space-2: 8px;  --space-3: 12px;
--space-4: 16px; --space-6: 24px;
--radius-panel: 4px;  --radius-chip: 3px;   /* 小圆角，Grafana 系不用大圆角 */

--row-h: 28px;          /* 表格行高：密度的核心参数 */
--panel-title-h: 32px;
--rail-w: 64px;  --rail-w-expanded: 220px;  --rail-icon: 20px;  --topbar-h: 48px;
--fs-xs: 11px; --fs-sm: 12px; --fs-md: 13px; --fs-lg: 15px; --fs-xl: 20px;
```

**密度基准**：1440×900 的屏幕上，Fleet 页应能同时看到 ≥6 个面板、≥20 行表格数据，
不需要滚动即可回答"现在哪个模型占着显存、最近一小时吞吐多少、有没有异常"。

---

## 3. 布局骨架

```
┌──┬──────────────────────────────────────────────────────────┐
│  │ Topbar 48px: 标题 · 副标题 ······························ ☾ 主题 │
│R ├──────────────────────────────────────────────────────────┤
│a │  StatCard 行（4–6 个，等高）                                │
│i │  ┌────────────────────┬───────────────────────────────┐   │
│l │  │ Panel（12 列网格）  │ Panel                         │   │
│64│  ├────────────────────┴───────────────────────────────┤   │
│  │  │ DataTable（密集行、内联 sparkline、游标分页）         │   │
│  │  └────────────────────────────────────────────────────┘   │
└──┴──────────────────────────────────────────────────────────┘
```
- 左 rail：默认 64px 只放图标；**点 logo ◈ 切换** 64 ↔ 220px（◈ 与下面的菜单图标同列同字号，
  不额外占一行），开合状态存 localStorage（`onyx.rail-open`）。不做悬停展开——文字随机出现会让
  "这一项到底能不能点"变得不可预期，而读数时不该有这种惊喜
- 侧栏里**不放说明文字**：展开态只有图标 + 菜单名 + 评测的四个子页；收起态子页回落到内容区顶部的
  subnav，否则矩阵与回归又只剩手敲 hash 这一条路
- rail 底部常驻引擎状态块（`onyx.engine-open`）：rail 收起时只剩状态灯，点它即展开侧栏；
  展开时给出 `provider_id · provider_kind`、引擎/schema 版本、base_url、数据新鲜度。
  结论词（可达 / 不可达 / 连接中…）与灯同时在场——"不可达"不该藏在一跳之后；
  还没拿到数据用灰点，红点只留给确实不可达（R2/R5）
- 主区 12 列网格，gap 12px；面板可折叠，折叠状态存 localStorage
- 面板标题行是 `min-height: 32px` + 允许换行，不是固定高度：窄面板上「prefill 冷 / 热」这类
  两行标题会把固定行高顶破、字溢出边框外。同理 `.btn` 一律 `white-space: nowrap`，
  否则"刷新"能被挤成竖排两个字；时序行尾的「峰值 / 最新 / 均值 t/s」这类短标签也带 `nowrap`，
  被挤时会一字一行竖着排，而它是那行数字的单位
- `flush`（去掉面板内缩）**只给表格用**——单元格自带 8px 横向 padding。表单、统计行、说明文字
  放进 flush 面板就是"字贴着边框"，这类面板要留默认的 12px 内缩
- 路由用 hash（`#/fleet`、`#/traces/:id`），不引第三方 router

---

## 4. 组件清单

| 组件 | 关键行为 |
|---|---|
| `Panel` | 标题左、口径说明右（如 `in_tokens · source=engine`）、可折叠、右上角操作区 |
| `StatCard` | 大数字（`--fs-xl` + mono + tabular）+ 标签 + 环比 delta + 内联 sparkline；未知显示「—」 |
| `SourceBadge` | `engine`/`fitted`/`heuristic` 三种底色 + `high/medium/low` 点；low 加虚线下划线与 tooltip |
| `PrefillTag` | `cold`(蓝)/`warm`(橙)/`unknown`(灰)，带 tooltip 解释 P11 |
| `DataTable` | 行高 28px、斑马纹关闭（用 1px 分隔线）、列头可排序、数字右对齐、行点击进详情 |
| `Sparkline` | 纯 SVG，无依赖；cold/warm 两条系列分开画 |
| `AnomalyChip` | 码 + 严重度色 + hover 显示 `meaning`/`action`（文案来自后端码表，前后端不各写一份） |
| `TraceTimeline` | 横向条：load → prompt_eval → decode，各段按真实毫秒比例；点击段看数字 |
| `PromptBreakdown` | **堆叠条**（不是饼图）：system / tool_defs / msg:i / template_ctl，hover 显示 token 与占比 |
| `RawJsonViewer` | 等宽、可折叠、带"复制为 curl"；只读 |
| `CommandPalette` | ⌘K：跳页面、按 id 打开 trace、切模型 |
| `EmptyState` / `Skeleton` | 空态写清"为什么空"与"下一步做什么"，不只是"暂无数据" |

---

## 5. 明确不做（避免跑偏）

- ❌ 大圆角卡片 + 大留白的营销风：同屏信息量不够，排障要来回滚
- ❌ 饼图表示 token 分布：占比接近时分不出来，用堆叠条
- ❌ 亮色主题为默认：长时间盯盘刺眼（保留亮色作为可切换项）
- ❌ 前端自行计算任何指标：所有派生值由后端给出（口径唯一）
- ❌ 显示没有 `source` 的数字（R1）
- ❌ 图表动画超过 150ms：数据刷新时动画会掩盖变化

---

## 6. 技术选型

**Vite + React 18 + TypeScript + 手写 CSS（设计 token 驱动）**，不引 Tailwind / 组件库。

理由：
1. Grafana 式高密度看板的价值在**像素级控制**（28px 行高、1px 分隔线、tabular-nums 对齐），
   通用组件库的默认密度总是偏松，改起来比自己写还慢；
2. 依赖面越小，`npm install` 与构建越可复现（这台机器网络有超时史）；
3. 设计 token 集中在一个 CSS 文件里就是设计系统本身，换主题只改一处。

代价：表格虚拟滚动、下拉定位这类要自己写。当前数据量（单机、万级 trace）用游标分页足够，
真需要虚拟滚动时再引 `@tanstack/react-virtual` 一个包。

图表用**手写 SVG**（sparkline / 堆叠条 / 时间轴都是几十行的形状），不引 recharts/echarts：
这些图的形态是固定的，引一个 400KB 的图表库换不来灵活性，还会带来默认样式与设计语言冲突。

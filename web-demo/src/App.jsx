import { Fragment, useEffect, useMemo, useState } from "react";

const tabs = [
  { id: "graph", label: "Graph" },
  { id: "stats", label: "Thống kê" },
];

const numberFormatter = new Intl.NumberFormat("vi-VN");
const percentFormatter = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 1 });
const chartPalette = [
  "#2563eb",
  "#059669",
  "#d97706",
  "#dc2626",
  "#7c3aed",
  "#0891b2",
  "#be123c",
  "#4d7c0f",
];

function publicUrl(path) {
  return `${import.meta.env.BASE_URL}${String(path).replace(/^\//, "")}`;
}

function formatNumber(value) {
  return numberFormatter.format(value ?? 0);
}

function formatScore(value, digits = 4) {
  return Number(value ?? 0).toFixed(digits);
}

function clamp(value, min = 0, max = 100) {
  return Math.min(max, Math.max(min, value));
}

function buildScaleTicks(max, formatter = formatNumber, minScale = 1) {
  const safeMax = Math.max(Number(max ?? 0), minScale);
  return [0, 0.25, 0.5, 0.75, 1].map((ratio) => ({
    ratio,
    label: formatter(safeMax * ratio),
  }));
}

function formatBytes(value) {
  const bytes = Number(value ?? 0);
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${formatNumber(bytes)} B`;
}

function App() {
  const [activeTab, setActiveTab] = useState("graph");
  const graphIndexState = useGraphIndex();

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Phân tích dữ liệu Subreddit</h1>
        </div>
      </header>

      <nav className="tabs" aria-label="Điều hướng demo">
        {tabs.map((tab) => (
          <button
            className={activeTab === tab.id ? "active" : ""}
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main>
        {activeTab === "graph" ? (
          <GraphTab graphIndexState={graphIndexState} />
        ) : (
          <StatsTab />
        )}
      </main>
    </div>
  );
}

function useGraphIndex() {
  const [state, setState] = useState({ graphs: [], loading: true, error: "" });

  useEffect(() => {
    let cancelled = false;

    fetch(publicUrl("graphs/index.json"))
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Không tải được graphs/index.json (${response.status})`);
        }
        return response.json();
      })
      .then((payload) => {
        if (!cancelled) {
          setState({ graphs: payload.graphs ?? [], loading: false, error: "" });
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setState({ graphs: [], loading: false, error: error.message });
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}

function SummaryPill({ label, value }) {
  return (
    <div className="summary-pill">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function GraphTab({ graphIndexState }) {
  return <GraphHtmlTab graphIndexState={graphIndexState} />;
}

function GraphHtmlTab({ graphIndexState }) {
  const { graphs, loading, error } = graphIndexState;
  const [selectedGraphId, setSelectedGraphId] = useState("");
  const [cachedGraphIds, setCachedGraphIds] = useState([]);

  useEffect(() => {
    if (selectedGraphId || graphs.length === 0) return;
    const initialGraph = graphs.find((graph) => graph.id === "community_level") ?? graphs[0];
    setSelectedGraphId(initialGraph.id);
  }, [graphs, selectedGraphId]);

  const selectedMeta = useMemo(
    () => (selectedGraphId ? graphs.find((graph) => graph.id === selectedGraphId) ?? null : null),
    [graphs, selectedGraphId],
  );
  const selectedPath = selectedMeta?.htmlPath ?? selectedMeta?.path ?? "";
  const cachedGraphs = useMemo(
    () => graphs.filter((graph) => cachedGraphIds.includes(graph.id)),
    [cachedGraphIds, graphs],
  );

  useEffect(() => {
    if (!selectedGraphId || !selectedPath) return;
    setCachedGraphIds((current) =>
      current.includes(selectedGraphId) ? current : [...current, selectedGraphId],
    );
  }, [selectedGraphId, selectedPath]);

  if (loading) {
    return <EmptyState title="Đang tải danh sách graph" text="Metadata nhỏ đang được đọc trước." />;
  }

  if (error) {
    return <EmptyState title="Không tải được danh sách graph" text={error} />;
  }

  if (!graphs.length) {
    return <EmptyState title="Chưa có graph" text="Không tìm thấy file HTML graph để hiển thị." />;
  }

  return (
    <section className="graph-workspace graph-workspace-html">
      <aside className="graph-list" aria-label="Danh sách graph">
        {graphs.map((graph) => (
          <button
            className={graph.id === selectedGraphId ? "graph-item active" : "graph-item"}
            key={graph.id}
            onClick={() => setSelectedGraphId(graph.id)}
            type="button"
          >
            <span>{graph.title}</span>
            <strong>{getGraphItemMeta(graph)}</strong>
          </button>
        ))}
      </aside>

      <section className="graph-panel graph-panel-html">
        <div className="graph-header">
          <div>
            <h2>{selectedMeta?.title ?? "Graph"}</h2>
            <p>{selectedMeta?.description}</p>
          </div>
        </div>

        {selectedPath ? (
          <div className="graph-frame-shell">
            {cachedGraphs.map((graph) => {
              const graphPath = graph.htmlPath ?? graph.path ?? "";
              const active = graph.id === selectedGraphId;

              return (
                <iframe
                  aria-hidden={!active}
                  className={active ? "graph-frame active" : "graph-frame"}
                  key={graph.id}
                  src={publicUrl(graphPath)}
                  tabIndex={active ? undefined : -1}
                  title={graph.title ?? "Graph"}
                />
              );
            })}
          </div>
        ) : (
          <div className="inline-message error">Graph này chưa có đường dẫn HTML.</div>
        )}
      </section>
    </section>
  );
}

function getGraphItemMeta(graph) {
  const parts = [];
  if (Number(graph.nodeCount) > 0) parts.push(`${formatNumber(graph.nodeCount)} node`);
  if (Number(graph.edgeCount) > 0) parts.push(`${formatNumber(graph.edgeCount)} cạnh`);
  if (graph.sourceFile) parts.push(graph.sourceFile);
  if (!parts.length && graph.path) parts.push(graph.path);
  return parts.join(", ");
}

function StatsTab() {
  const [state, setState] = useState({ data: null, loading: true, error: "" });

  useEffect(() => {
    let cancelled = false;

    fetch(publicUrl("data/demoData.json"))
      .then((response) => {
        if (!response.ok) throw new Error(`Không tải được demoData.json (${response.status})`);
        return response.json();
      })
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: "" });
      })
      .catch((error) => {
        if (!cancelled) setState({ data: null, loading: false, error: error.message });
      });

    return () => {
      cancelled = true;
    };
  }, []);

  if (state.loading) {
    return <EmptyState title="Đang tải bảng thống kê" text="Dữ liệu bảng được tải riêng khỏi tab Graph." />;
  }

  if (state.error) {
    return <EmptyState title="Không tải được dữ liệu thống kê" text={state.error} />;
  }

  const data = state.data;
  const strongEdgePercent =
    (Number(data.keyMetrics.strongEdges ?? 0) / Math.max(Number(data.keyMetrics.similarityPairs ?? 0), 1)) *
    100;
  const metricCards = [
    {
      label: "Subreddits",
      value: data.keyMetrics.subreddits,
      display: formatNumber(data.keyMetrics.subreddits),
      helper: "Node trong graph semantic",
      maxValue: data.keyMetrics.subreddits,
      color: chartPalette[0],
    },
    {
      label: "Communities",
      value: data.keyMetrics.communities,
      display: formatNumber(data.keyMetrics.communities),
      helper: "Cụm Louvain sau phân rã",
      maxValue: data.keyMetrics.subreddits,
      color: chartPalette[1],
    },
    {
      label: "Similarity pairs",
      value: data.keyMetrics.similarityPairs,
      display: formatNumber(data.keyMetrics.similarityPairs),
      helper: "Cặp có similarity >= 0,3",
      maxValue: data.keyMetrics.similarityPairs,
      color: chartPalette[5],
    },
    {
      label: "Cạnh top 3%",
      value: data.keyMetrics.strongEdges,
      display: formatNumber(data.keyMetrics.strongEdges),
      helper: `${percentFormatter.format(strongEdgePercent)}% tổng số cặp`,
      maxValue: data.keyMetrics.similarityPairs,
      color: chartPalette[2],
    },
    {
      label: "Ngưỡng P97",
      value: data.keyMetrics.topPercentileThreshold,
      display: formatScore(data.keyMetrics.topPercentileThreshold, 4),
      helper: "Threshold edge mạnh",
      maxValue: 1,
      color: chartPalette[4],
    },
    {
      label: "Modularity",
      value: data.keyMetrics.modularity,
      display: formatScore(data.keyMetrics.modularity, 4),
      helper: "Chất lượng community",
      maxValue: 1,
      color: chartPalette[3],
    },
  ];
  const communityDistribution = data.communitySizeDistribution.map((item) => ({
    label: `${item.label}`,
    value: item.count,
    detail: `${formatNumber(item.count)} community`,
  }));
  const similaritySummary = [
    { label: "Mean", value: data.similarityStats.mean },
    { label: "Median", value: data.similarityStats.p50 },
    { label: "P97", value: data.similarityStats.p97 },
    { label: "Strong edges", value: data.similarityStats.strongEdgeCount, format: formatNumber },
  ];
  const similarityQuantiles = [
    { label: "Min", value: data.similarityStats.min },
    { label: "Median", value: data.similarityStats.p50 },
    { label: "P75", value: data.similarityStats.p75 },
    { label: "P90", value: data.similarityStats.p90 },
    { label: "P97", value: data.similarityStats.p97 },
    { label: "P99", value: data.similarityStats.p99 },
    { label: "Max", value: data.similarityStats.max },
  ];
  const topSimilarityPairs = data.topSimilarityPairs.slice(0, 14).map((pair) => ({
    label: `${pair.source} - ${pair.target}`,
    value: pair.score,
    detail: formatScore(pair.score, 4),
    tooltip: formatPairContentTooltip(pair),
  }));
  const subredditContentRows = (data.subredditContentTable ?? []).slice(0, 24).map((row) => [
    row.subreddit,
    row.content,
    formatContentSource(row.source),
    row.communityName,
  ]);
  const topCommunities = data.topCommunities.slice(0, 12).map((community) => ({
    label: community.name,
    value: community.size,
    detail: `${formatNumber(community.size)} subreddit`
  }));
  const topBridges = data.topBridges.slice(0, 12).map((row) => ({
    label: row.subreddit,
    value: row.bridge_score,
    detail: formatScore(row.bridge_score, 4),
    subtext: `${row.source_name} -> ${row.target_name}`,
  }));
  const topBridgeMax = Math.max(...topBridges.map((item) => Number(item.value ?? 0)), 0);
  const topBridgeAxisMax = topBridgeMax > 0 ? topBridgeMax * 1.05 : 0;
  const topGateways = data.topGateways.slice(0, 12).map((row) => ({
    label: row.subreddit,
    value: row.gateway_score_normalized,
    detail: formatScore(row.gateway_score_normalized, 4),
    subtext: row.community_name,
  }));
  const topHighways = data.topHighways.slice(0, 12).map((row) => ({
    label: compactPath(row.pathNodes),
    value: row.occurrence_count,
    detail: `${formatNumber(row.occurrence_count)} lần`,
    subtext: `${row.unique_communities_spanned} community`,
  }));
  const runtimeSeries = [
    { key: "extract", label: "Extract", color: chartPalette[0] },
    { key: "embedding", label: "Embedding", color: chartPalette[1] },
    { key: "similarity", label: "Similarity", color: chartPalette[2] },
  ];
  const runtimeGroups = data.experiments.environments.map((row) => ({
    label: row.name,
    note: `${row.data} / ${row.ram}`,
    values: {
      extract: parseDurationMinutes(row.extract),
      embedding: parseDurationMinutes(row.embedding),
      similarity: parseDurationMinutes(row.similarity),
    },
    details: {
      extract: row.extract,
      embedding: row.embedding,
      similarity: row.similarity,
    },
  }));
  const louvainAlgorithm = data.experiments.algorithms.find((row) => row.name === "Louvain");
  const louvainTimeMinutes = parseDurationMinutes(louvainAlgorithm?.time);
  const algorithmRows = data.experiments.algorithms.map((row) => ({
    label: row.name,
    timeMinutes: parseDurationMinutes(row.time),
    detail: row.time,
    modularity: row.modularity,
    note: row.note,
    scope: row.scope,
    isFocus: row.name === "Louvain",
    deltaModularity: row.modularity - Number(louvainAlgorithm?.modularity ?? row.modularity),
    timeRatio:
      louvainTimeMinutes > 0 ? parseDurationMinutes(row.time) / louvainTimeMinutes : null,
  }));

  return (
    <section className="stats-stack stats-dashboard">
      <div className="section-title stats-title">
        <h2>Kết quả khai phá dữ liệu</h2>
      </div>

      <div className="metric-grid metric-chart-grid">
        {metricCards.map((metric) => (
          <MetricChartCard key={metric.label} metric={metric} />
        ))}
      </div>

      <div className="stats-grid stats-grid-featured">
        <Panel eyebrow="Histogram" title="Phân phối kích thước community">
          <VerticalBarChart
            data={communityDistribution}
            xLabel="Số subreddit / community"
            yLabel="Số community"
            valueFormatter={formatNumber}
          />
        </Panel>
        <Panel eyebrow="Quantile" title="Phân phối similarity">
          <div className="stat-strip">
            {similaritySummary.map((item) => (
              <div className="stat-strip-item" key={item.label}>
                <span>{item.label}</span>
                <strong>{(item.format ?? ((value) => formatScore(value, 4)))(item.value)}</strong>
              </div>
            ))}
          </div>
          <QuantileChart
            data={similarityQuantiles}
            mean={data.similarityStats.mean}
            min={0}
            max={1}
            threshold={data.keyMetrics.topPercentileThreshold}
            valueFormatter={(value) => formatScore(value, 4)}
          />
        </Panel>
      </div>

      <Panel eyebrow="Ranking score" title="Top similarity pairs">
        <ColumnBarChart
          data={topSimilarityPairs}
          maxValue={1}
          axisFormatter={(value) => formatScore(value, 2)}
          valueFormatter={(value) => formatScore(value, 4)}
        />
      </Panel>

      <div className="stats-grid">
        <Panel eyebrow="Ranking size" title="Top community lớn">
          <ColumnBarChart data={topCommunities} valueFormatter={formatNumber} />
        </Panel>
        <Panel eyebrow="Role overlap" title="Bridge và Gateway">
          <RoleSummaryChart
            bridgeRows={data.keyMetrics.bridgeRows}
            gatewayRows={data.keyMetrics.gatewayRows}
            overlap={data.bridgeGatewayOverlap}
          />
        </Panel>
      </div>

      <div className="stats-grid">
        <Panel eyebrow="Betweenness role" title="Top bridge">
          <ColumnBarChart
            data={topBridges}
            maxValue={topBridgeAxisMax}
            minScale={0}
            axisFormatter={(value) => formatScore(value, 3)}
            valueFormatter={(value) => formatScore(value, 4)}
          />
        </Panel>
        {/* <Panel eyebrow="Gateway role" title="Top gateway">
          <ColumnBarChart
            data={topGateways}
            axisFormatter={(value) => formatScore(value, 2)}
            maxValue={1}
            valueFormatter={(value) => formatScore(value, 4)}
          />
        </Panel> */}
      </div>

      <div className="stats-grid">
        <Panel eyebrow="Path frequency" title="Top highway">
          <ColumnBarChart data={topHighways} valueFormatter={formatNumber} />
        </Panel>
        <Panel eyebrow="Matrix" title="Highway heatmap">
          <HighwayHeatmap data={data.highwayHeatmap} />
        </Panel>
      </div>

      <div className="stats-grid">
        <Panel eyebrow="Grouped runtime" title="Thời gian xử lý">
          <GroupedBarChart
            groups={runtimeGroups}
            series={runtimeSeries}
            valueFormatter={(value) => `${formatScore(value, 1)} phút`}
          />
        </Panel>
        <Panel eyebrow="Quality / runtime" title="Thuật toán community">
          <AlgorithmBenchmarkChart
            data={algorithmRows}
            focusLabel="Louvain"
            timeFormatter={(value) => `${formatScore(value, 1)} phút`}
          />
        </Panel>
      </div>
    </section>
  );
}

function compactPath(nodes) {
  return (nodes ?? []).join(" → ");
}

function formatPairContentTooltip(pair) {
  const source = formatSubredditContentTooltip(pair.source, pair.sourceContent);
  const target = formatSubredditContentTooltip(pair.target, pair.targetContent);
  return [`${pair.source} - ${pair.target}`, `Similarity: ${formatScore(pair.score, 4)}`, source, target]
    .filter(Boolean)
    .join("\n\n");
}

function formatSubredditContentTooltip(subreddit, content) {
  if (!content) return `${subreddit}: chưa có nội dung`;
  const source = formatContentSource(content.source);
  const body = content.shortContent || content.content || "Chưa có nội dung";
  return `${subreddit} (${source}): ${body}`;
}

function formatContentSource(source) {
  const normalized = String(source ?? "").trim().toLowerCase();
  if (normalized === "reddit_about") return "Reddit about.json";
  if (normalized === "community_inference") return "Suy luận từ community";
  if (!normalized) return "Không rõ nguồn";
  return source;
}

function MetricChartCard({ metric }) {
  const value = Number(metric.value ?? 0);
  const max = Math.max(Number(metric.maxValue ?? value), 1);
  const percent = clamp((value / max) * 100);

  return (
    <article className="metric-chart-card" style={{ "--metric-color": metric.color }}>
      <span>{metric.label}</span>
      <strong>{metric.display}</strong>
      <div className="metric-meter" title={`${metric.label}: ${metric.display}`}>
        <div style={{ width: `${percent}%` }} />
      </div>
      <small>{metric.helper}</small>
    </article>
  );
}

function ColumnBarChart({
  data,
  maxValue,
  ranked = true,
  valueFormatter = formatNumber,
  axisFormatter = valueFormatter,
  minScale = 1,
}) {
  const max = maxValue ?? Math.max(...data.map((item) => Number(item.value ?? 0)), 1);
  const ticks = buildScaleTicks(max, axisFormatter, minScale);

  return (
    <div className="column-chart">
      <div
        className="column-frame"
        style={{
          "--column-count": data.length,
          minWidth: `${Math.max(480, data.length * 78 + 62)}px`,
        }}
      >
        <div className="column-y-axis">
          {[...ticks].reverse().map((tick) => (
            <span key={tick.ratio}>{tick.label}</span>
          ))}
        </div>
        <div className="column-plot">
          <div className="column-grid-lines" />
          <div className="column-bars">
            {data.map((item, index) => {
              const value = Number(item.value ?? 0);
              const height = max > 0 ? clamp((value / max) * 100, value > 0 ? 3 : 0, 100) : 0;
              const color = item.color ?? chartPalette[index % chartPalette.length];
              const valueLabel = item.detail ?? valueFormatter(value);
              const tooltip = item.tooltip ?? `${item.label}: ${valueFormatter(value)}`;

              return (
                <div
                  aria-label={tooltip}
                  className="column-item"
                  key={item.label}
                  style={{ "--bar-color": color }}
                  tabIndex={item.tooltip ? 0 : undefined}
                >
                  <strong className="column-value">{valueLabel}</strong>
                  <div className="column-bar-shell" title={tooltip}>
                    <div className="column-bar" style={{ height: `${height}%` }} />
                  </div>
                  {item.tooltip && <div className="column-tooltip">{item.tooltip}</div>}
                  <div className="column-label">
                    <strong title={item.label}>{item.label}</strong>
                    {item.subtext && <small title={item.subtext}>{item.subtext}</small>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

function VerticalBarChart({ data, maxValue, valueFormatter = formatNumber, xLabel, yLabel }) {
  const max = maxValue ?? Math.max(...data.map((item) => Number(item.value ?? 0)), 1);
  const ticks = [max, Math.ceil(max / 2), 0];

  return (
    <div className="vbar-chart">
      {yLabel && <div className="vbar-axis-title">{yLabel}</div>}
      <div className="vbar-frame">
        <div className="vbar-y-axis">
          {ticks.map((tick) => (
            <span key={tick}>{valueFormatter(tick)}</span>
          ))}
        </div>
        <div className="vbar-plot">
          <div className="vbar-grid-lines" />
          <div className="vbar-bars">
            {data.map((item, index) => {
              const value = Number(item.value ?? 0);
              const height = max > 0 ? clamp((value / max) * 100, value > 0 ? 3 : 0, 100) : 0;
              const color = item.color ?? chartPalette[index % chartPalette.length];

              return (
                <div className="vbar-item" key={item.label} style={{ "--bar-color": color }}>
                  <strong>{item.detail ?? valueFormatter(value)}</strong>
                  <div className="vbar-column" title={`${item.label}: ${valueFormatter(value)}`}>
                    <div style={{ height: `${height}%` }} />
                  </div>
                  <span>{item.label}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
      {xLabel && <div className="vbar-x-label">{xLabel}</div>}
    </div>
  );
}

function QuantileChart({
  data,
  mean,
  threshold,
  min = 0,
  max = 1,
  valueFormatter = formatNumber,
}) {
  const range = Math.max(max - min, Number.EPSILON);
  const getLeft = (value) => clamp(((Number(value ?? 0) - min) / range) * 100);
  const median = data.find((item) => item.label === "Median")?.value;
  const p75 = data.find((item) => item.label === "P75")?.value;
  const p90 = data.find((item) => item.label === "P90")?.value;
  const p99 = data.find((item) => item.label === "P99")?.value;

  return (
    <div className="quantile-chart">
      <div className="quantile-ruler">
        <div className="quantile-axis" />
        {median != null && p75 != null && (
          <div
            className="quantile-band median-band"
            style={{ left: `${getLeft(median)}%`, width: `${getLeft(p75) - getLeft(median)}%` }}
            title={`Median - P75: ${valueFormatter(median)} - ${valueFormatter(p75)}`}
          />
        )}
        {p75 != null && p90 != null && (
          <div
            className="quantile-band high-band"
            style={{ left: `${getLeft(p75)}%`, width: `${getLeft(p90) - getLeft(p75)}%` }}
            title={`P75 - P90: ${valueFormatter(p75)} - ${valueFormatter(p90)}`}
          />
        )}
        {p90 != null && p99 != null && (
          <div
            className="quantile-band tail-band"
            style={{ left: `${getLeft(p90)}%`, width: `${getLeft(p99) - getLeft(p90)}%` }}
            title={`P90 - P99: ${valueFormatter(p90)} - ${valueFormatter(p99)}`}
          />
        )}
        {[min, 0.5, max].map((tick) => (
          <span className="quantile-tick" key={tick} style={{ left: `${getLeft(tick)}%` }}>
            {valueFormatter(tick)}
          </span>
        ))}
        {mean != null && (
          <span
            className="quantile-reference mean-reference"
            style={{ left: `${getLeft(mean)}%` }}
            title={`Mean: ${valueFormatter(mean)}`}
          >
            Mean
          </span>
        )}
        {threshold != null && (
          <span
            className="quantile-reference threshold-reference"
            style={{ left: `${getLeft(threshold)}%` }}
            title={`P97 threshold: ${valueFormatter(threshold)}`}
          >
            P97
          </span>
        )}
        {data.map((item, index) => {
          const value = Number(item.value ?? 0);
          const left = getLeft(value);
          const color = chartPalette[index % chartPalette.length];

          return (
            <button
              className="quantile-dot"
              key={item.label}
              style={{ left: `${left}%`, "--dot-color": color }}
              title={`${item.label}: ${valueFormatter(value)}`}
              type="button"
            >
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>
      <div className="quantile-cards">
        {data.map((item, index) => (
          <div
            className="quantile-card"
            key={item.label}
            style={{ "--dot-color": chartPalette[index % chartPalette.length] }}
          >
            <span>{item.label}</span>
            <strong>{valueFormatter(item.value)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function RoleSummaryChart({ bridgeRows, gatewayRows, overlap }) {
  const max = Math.max(Number(bridgeRows ?? 0), Number(gatewayRows ?? 0), 1);
  const rows = [
    { label: "Bridge rows", value: bridgeRows, color: "#dc2626" },
    { label: "Gateway rows", value: gatewayRows, color: "#059669" },
  ];

  return (
    <div className="role-chart">
      <div className="donut-row">
        <div className="donut" style={{ "--percent": overlap.percent }}>
          <strong>{percentFormatter.format(overlap.percent)}%</strong>
        </div>
        <div>
          <strong>{formatNumber(overlap.count)} overlap subreddit</strong>
          <p>Nằm đồng thời trong tập bridge và gateway nổi bật.</p>
        </div>
      </div>
      <ColumnBarChart
        axisFormatter={formatNumber}
        data={rows}
        maxValue={max}
        ranked={false}
        valueFormatter={formatNumber}
      />
    </div>
  );
}

function GroupedBarChart({ groups, series, valueFormatter = formatNumber }) {
  const max = Math.max(
    ...groups.flatMap((group) => series.map((item) => Number(group.values[item.key] ?? 0))),
    1,
  );
  const ticks = buildScaleTicks(max, valueFormatter);

  return (
    <div className="grouped-chart">
      <div className="chart-legend">
        {series.map((item) => (
          <span key={item.key} style={{ "--legend-color": item.color }}>
            {item.label}
          </span>
        ))}
      </div>
      <div
        className="grouped-column-frame"
        style={{
          "--group-count": groups.length,
          "--series-count": series.length,
          minWidth: `${Math.max(460, groups.length * 132 + 62)}px`,
        }}
      >
        <div className="grouped-y-axis">
          {[...ticks].reverse().map((tick) => (
            <span key={tick.ratio}>{tick.label}</span>
          ))}
        </div>
        <div className="grouped-column-plot">
          <div className="grouped-column-grid" />
          <div className="grouped-column-groups">
            {groups.map((group) => (
              <div className="grouped-column-group" key={group.label}>
                <div className="grouped-column-bars">
                  {series.map((item) => {
                    const value = Number(group.values[item.key] ?? 0);
                    const height = clamp((value / max) * 100, value > 0 ? 3 : 0, 100);
                    return (
                      <div className="grouped-column-item" key={item.key}>
                        <strong>{group.details[item.key]}</strong>
                        <div
                          className="grouped-column-track"
                          title={`${group.label} - ${item.label}: ${group.details[item.key]}`}
                        >
                          <div style={{ height: `${height}%`, backgroundColor: item.color }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="grouped-column-label">
                  <strong>{group.label}</strong>
                  <span>{group.note}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function AlgorithmBenchmarkChart({ data, focusLabel = "Louvain", timeFormatter = formatNumber }) {
  const focus = data.find((item) => item.label === focusLabel) ?? data[0];
  const focusTime = Math.max(Number(focus?.timeMinutes ?? 1), Number.EPSILON);
  const focusModularity = Number(focus?.modularity ?? 0);
  const rows = data.map((item) => ({
    ...item,
    timeRatio: item.timeRatio ?? Number(item.timeMinutes ?? 0) / focusTime,
  }));
  const maxLogRatio = Math.max(
    ...rows.map((item) => Math.abs(Math.log10(Math.max(Number(item.timeRatio ?? 1), Number.EPSILON)))),
    0.5,
  );
  const minLogRatio = -maxLogRatio;
  const modularityPadding =
    Math.max(...rows.map((item) => Math.abs(Number(item.modularity ?? 0) - focusModularity)), 0.02) + 0.02;
  const minModularity = focusModularity - modularityPadding;
  const maxModularity = focusModularity + modularityPadding;
  const modularityRange = Math.max(maxModularity - minModularity, Number.EPSILON);
  const plot = { left: 72, right: 604, top: 28, bottom: 206 };
  const width = plot.right - plot.left;
  const height = plot.bottom - plot.top;
  const xScale = (ratio) => {
    const logRatio = Math.log10(Math.max(Number(ratio ?? 1), Number.EPSILON));
    return plot.left + ((logRatio - minLogRatio) / (maxLogRatio - minLogRatio)) * width;
  };
  const yScale = (value) =>
    plot.bottom - ((Number(value ?? 0) - minModularity) / modularityRange) * height;
  const xTicks = [0.01, 0.1, 1, 10, 100, 1000].filter((ratio) => {
    const logRatio = Math.log10(ratio);
    return logRatio >= minLogRatio && logRatio <= maxLogRatio;
  });
  const yTicks = [minModularity, focusModularity, maxModularity].filter(
    (value, index, values) => values.findIndex((candidate) => Math.abs(candidate - value) < 0.0001) === index,
  );
  const focusX = xScale(1);
  const focusY = yScale(focusModularity);

  return (
    <div className="algorithm-chart">
      <svg aria-label="Benchmark thuật toán community" viewBox="0 0 640 250">
        <line className="chart-axis" x1={plot.left} x2={plot.left} y1={plot.top} y2={plot.bottom} />
        <line className="chart-axis" x1={plot.left} x2={plot.right} y1={plot.bottom} y2={plot.bottom} />
        {xTicks.map((tick) => (
          <g key={`x-${tick}`}>
            <line className="chart-grid-line" x1={xScale(tick)} x2={xScale(tick)} y1={plot.top} y2={plot.bottom} />
            <text className="chart-tick" textAnchor="middle" x={xScale(tick)} y={plot.bottom + 24}>
              {formatRatio(tick)}
            </text>
          </g>
        ))}
        {yTicks.map((tick) => (
          <g key={`y-${tick}`}>
            <line className="chart-grid-line" x1={plot.left} x2={plot.right} y1={yScale(tick)} y2={yScale(tick)} />
            <text className="chart-tick" textAnchor="end" x={plot.left - 12} y={yScale(tick) + 4}>
              {formatScore(tick, 2)}
            </text>
          </g>
        ))}
        {focus && (
          <>
            <line className="chart-focus-line" x1={focusX} x2={focusX} y1={plot.top} y2={plot.bottom} />
            <line className="chart-focus-line" x1={plot.left} x2={plot.right} y1={focusY} y2={focusY} />
          </>
        )}
        <text className="chart-axis-label" textAnchor="middle" x={(plot.left + plot.right) / 2} y={246}>
          Thời gian so với Louvain
        </text>
        <text className="chart-axis-label" textAnchor="middle" transform="rotate(-90 20 116)" x={20} y={116}>
          Modularity
        </text>
        {rows.map((item, index) => {
          const color = getAlgorithmColor(item, index, focusLabel);
          const pointX = xScale(item.timeRatio);
          const pointY = yScale(item.modularity);
          const labelOnLeft = pointX > plot.right - 124;
          return (
            <g key={item.label}>
              <circle
                className={item.label === focusLabel ? "chart-point focus" : "chart-point"}
                cx={pointX}
                cy={pointY}
                fill={color}
                r={item.label === focusLabel ? "11" : "9"}
              />
              {item.label === focusLabel && (
                <circle className="chart-focus-ring" cx={pointX} cy={pointY} r="16" />
              )}
              <text
                className={item.label === focusLabel ? "chart-point-label focus" : "chart-point-label"}
                textAnchor={labelOnLeft ? "end" : "start"}
                x={pointX + (labelOnLeft ? -14 : 14)}
                y={pointY + 5}
              >
                {item.label}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="algorithm-cards">
        {rows.map((item, index) => (
          <div
            className={item.label === focusLabel ? "algorithm-card focus" : "algorithm-card"}
            key={item.label}
            style={{ "--metric-color": getAlgorithmColor(item, index, focusLabel) }}
          >
            <span>{item.label}</span>
            <strong>{item.detail}</strong>
            <small>
              {timeFormatter(item.timeMinutes)} / Modularity {formatScore(item.modularity, 4)}
              {item.scope === "local" ? " (local)" : ""}
            </small>
            {item.note && <small>{item.note}</small>}
          </div>
        ))}
      </div>
    </div>
  );
}

function getAlgorithmColor(item, index, focusLabel) {
  if (item.label === focusLabel) return "#2563eb";
  if (item.label.toLowerCase().includes("pagerank")) return "#7c3aed";
  return chartPalette[(index + 2) % chartPalette.length];
}

function formatRatio(value) {
  const ratio = Number(value ?? 1);
  if (!Number.isFinite(ratio)) return "n/a";
  if (ratio < 0.1) return `${formatScore(ratio, 2)}x`;
  if (ratio < 10) return `${formatScore(ratio, 1)}x`;
  return `${formatScore(ratio, 0)}x`;
}

function formatSignedScore(value) {
  const number = Number(value ?? 0);
  const prefix = number > 0 ? "+" : "";
  return `${prefix}${formatScore(number, 4)}`;
}

function parseDurationMinutes(value) {
  const text = String(value ?? "").toLowerCase();
  let minutes = 0;
  const toNumber = (raw) => Number(String(raw).replace(",", "."));
  const hours = text.match(/(\d+(?:[.,]\d+)?)\s*h/);
  const seconds = text.match(/(\d+(?:[.,]\d+)?)\s*s/);
  const minuteMatches = [...text.matchAll(/(\d+(?:[.,]\d+)?)\s*(?:p|m)(?![a-z])/g)];

  if (hours) minutes += toNumber(hours[1]) * 60;
  for (const match of minuteMatches) minutes += toNumber(match[1]);
  if (seconds) minutes += toNumber(seconds[1]) / 60;

  if (minutes === 0 && Number.isFinite(Number(text))) return Number(text);
  return minutes;
}

function Panel({ title, eyebrow, children }) {
  return (
    <section className="panel">
      <header className="panel-heading">
        {eyebrow && <span>{eyebrow}</span>}
        <h3>{title}</h3>
      </header>
      {children}
    </section>
  );
}

function BarList({ data, valueLabel }) {
  const max = Math.max(...data.map((item) => item.value), 1);

  return (
    <div className="bar-list">
      {data.map((item) => (
        <div className="bar-row" key={item.label}>
          <div className="bar-label">
            <span>{item.label}</span>
            <strong>
              {formatNumber(item.value)} {valueLabel}
            </strong>
          </div>
          <div className="bar-track">
            <div style={{ width: `${Math.max(4, (item.value / max) * 100)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function HighwayHeatmap({ data }) {
  const maxOccurrence = Math.max(...data.map((item) => item.occurrence), 1);
  const lengths = [...new Set(data.map((item) => item.length))];
  const communityCounts = [...new Set(data.map((item) => item.communities))];

  return (
    <div className="heatmap-wrap">
      <div
        className="heatmap"
        style={{
          gridTemplateColumns: `92px repeat(${communityCounts.length}, minmax(58px, 1fr))`,
        }}
      >
        <div className="heatmap-label">Length</div>
        {communityCounts.map((count) => (
          <div className="heatmap-label" key={`head-${count}`}>
            {count} community
          </div>
        ))}
        {lengths.map((length) => (
          <Fragment key={`row-${length}`}>
            <div className="heatmap-label">{length} node</div>
            {communityCounts.map((count) => {
              const cell = data.find(
                (item) => item.length === length && item.communities === count,
              );
              const intensity = cell ? Math.max(0.12, cell.occurrence / maxOccurrence) : 0;
              return (
                <div
                  className="heatmap-cell"
                  key={`${length}-${count}`}
                  style={{ backgroundColor: `rgba(37, 99, 235, ${intensity})` }}
                  title={`Length ${length}, ${count} community: ${cell ? formatNumber(cell.occurrence) : 0}`}
                >
                  {cell ? formatNumber(cell.occurrence) : ""}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
      <div className="heatmap-legend">
        <span>Ít</span>
        <div />
        <span>Nhiều</span>
      </div>
    </div>
  );
}

function DataTable({ columns, rows, emptyText = "Không có dữ liệu." }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length}>{emptyText}</td>
            </tr>
          ) : (
            rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => (
                  <td key={`${rowIndex}-${cellIndex}`}>{cell}</td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

function EmptyState({ title, text }) {
  return (
    <section className="empty-state">
      <h2>{title}</h2>
      <p>{text}</p>
    </section>
  );
}

export default App;

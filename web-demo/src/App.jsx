import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DataSet, Network } from "vis-network/standalone/esm/vis-network";
import "vis-network/styles/vis-network.css";

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

function buildScaleTicks(max, formatter = formatNumber) {
  const safeMax = Math.max(Number(max ?? 0), 1);
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
  const graphTotals = useMemo(() => {
    const graphs = graphIndexState.graphs;
    return {
      graphs: graphs.length,
      nodes: graphs.reduce((sum, graph) => sum + Number(graph.nodeCount ?? 0), 0),
      edges: graphs.reduce((sum, graph) => sum + Number(graph.edgeCount ?? 0), 0),
    };
  }, [graphIndexState.graphs]);

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

function useCommunityIndex(enabled) {
  const [state, setState] = useState({ data: null, loading: false, error: "" });

  useEffect(() => {
    if (!enabled || state.data || state.loading) return undefined;

    let cancelled = false;
    setState((current) => ({ ...current, loading: true, error: "" }));

    fetch(publicUrl("data/communityIndex.json"))
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Không tải được communityIndex.json (${response.status})`);
        }
        return response.json();
      })
      .then((data) => {
        if (!cancelled) {
          setState({ data, loading: false, error: "" });
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setState({ data: null, loading: false, error: error.message });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [enabled, state.data]);

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
  const { graphs, loading, error } = graphIndexState;
  const [selectedGraphId, setSelectedGraphId] = useState("");
  const [graphState, setGraphState] = useState({ data: null, loading: false, error: "" });
  const [groupFilter, setGroupFilter] = useState("all");
  const [showLabels, setShowLabels] = useState(false);
  const [nodeQuery, setNodeQuery] = useState("");
  const [selectedNode, setSelectedNode] = useState(null);
  const [pendingFocusId, setPendingFocusId] = useState("");
  const communityIndexState = useCommunityIndex(Boolean(selectedNode));
  const graphCacheRef = useRef(new Map());
  const prefetchingRef = useRef(new Set());
  const pendingFocusRef = useRef("");
  const containerRef = useRef(null);
  const networkRef = useRef(null);

  useEffect(() => {
    pendingFocusRef.current = pendingFocusId;
  }, [pendingFocusId]);

  useEffect(() => {
    if (selectedGraphId || graphs.length === 0) return;
    const initialGraph =
      graphs.find((graph) => graph.id === "community_level") ??
      [...graphs].sort((a, b) => Number(a.fileSizeBytes ?? 0) - Number(b.fileSizeBytes ?? 0))[0];
    setSelectedGraphId(initialGraph.id);
  }, [graphs, selectedGraphId]);

  const selectedMeta = useMemo(
    () => (selectedGraphId ? graphs.find((graph) => graph.id === selectedGraphId) ?? null : null),
    [graphs, selectedGraphId],
  );

  const prefetchGraph = useCallback((graph) => {
    if (!graph || graphCacheRef.current.has(graph.id) || prefetchingRef.current.has(graph.id)) {
      return;
    }
    prefetchingRef.current.add(graph.id);
    fetch(publicUrl(graph.path))
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (data) graphCacheRef.current.set(graph.id, data);
      })
      .catch(() => undefined)
      .finally(() => {
        prefetchingRef.current.delete(graph.id);
      });
  }, []);

  useEffect(() => {
    if (!selectedMeta) return undefined;

    const cached = graphCacheRef.current.get(selectedMeta.id);
    setSelectedNode(null);
    setPendingFocusId("");
    setGroupFilter("all");

    if (cached) {
      setGraphState({ data: cached, loading: false, error: "" });
      setShowLabels(cached.nodes.length <= 120);
      return undefined;
    }

    const controller = new AbortController();
    setGraphState({ data: null, loading: true, error: "" });

    fetch(publicUrl(selectedMeta.path), { signal: controller.signal })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Không tải được ${selectedMeta.path} (${response.status})`);
        }
        return response.json();
      })
      .then((data) => {
        graphCacheRef.current.set(selectedMeta.id, data);
        setGraphState({ data, loading: false, error: "" });
        setShowLabels(data.nodes.length <= 120);
      })
      .catch((fetchError) => {
        if (fetchError.name !== "AbortError") {
          setGraphState({ data: null, loading: false, error: fetchError.message });
        }
      });

    return () => {
      controller.abort();
    };
  }, [selectedMeta]);

  useEffect(() => {
    if (!graphs.length) return undefined;

    const smallGraphs = graphs.filter(
      (graph) => graph.id !== selectedGraphId && Number(graph.fileSizeBytes ?? 0) <= 160000,
    );
    const handle =
      "requestIdleCallback" in window
        ? window.requestIdleCallback(() => smallGraphs.forEach(prefetchGraph))
        : window.setTimeout(() => smallGraphs.forEach(prefetchGraph), 400);

    return () => {
      if ("cancelIdleCallback" in window) {
        window.cancelIdleCallback(handle);
      } else {
        window.clearTimeout(handle);
      }
    };
  }, [graphs, prefetchGraph, selectedGraphId]);

  const graphData = graphState.data;
  const groups = useMemo(() => graphData?.groups ?? [], [graphData]);
  const filteredGraph = useMemo(() => filterGraph(graphData, groupFilter), [graphData, groupFilter]);
  const visibleNodeIds = useMemo(
    () => new Set(filteredGraph.nodes.map((node) => node.id)),
    [filteredGraph.nodes],
  );

  useEffect(() => {
    if (!containerRef.current || !graphData) return undefined;

    let disposed = false;
    let idleHandle = null;
    let timerHandle = null;
    let network = null;
    const largeGraph = filteredGraph.nodes.length > 1000 || filteredGraph.edges.length > 1800;

    const renderNetwork = () => {
      if (disposed || !containerRef.current) return;

      const shouldShowLabels = showLabels || filteredGraph.nodes.length <= 80;
      const nodeSet = new DataSet(
        filteredGraph.nodes.map((node) => ({
          id: node.id,
          label: shouldShowLabels ? String(node.label ?? node.id) : undefined,
          title: normalizeTitle(node.title ?? node.label ?? node.id),
          group: String(node.group ?? "default"),
          shape: node.shape ?? "dot",
          size: node.size,
          value: node.value,
          color: node.color,
          x: node.x,
          y: node.y,
          borderWidth: 1,
        })),
      );
      const edgeSet = new DataSet(
        filteredGraph.edges.map((edge, index) => ({
          id: edge.id ?? `${edge.from}:${edge.to}:${index}`,
          from: edge.from,
          to: edge.to,
          title: normalizeTitle(edge.title ?? `${edge.from} - ${edge.to}`),
          width: Math.min(5, Math.max(0.6, Number(edge.width ?? 1))),
          value: edge.value,
          color: normalizeEdgeColor(edge.color),
          arrows: edge.arrows,
          dashes: edge.dashes,
          smooth: largeGraph ? false : { type: "continuous" },
        })),
      );

      network = new Network(
        containerRef.current,
        { nodes: nodeSet, edges: edgeSet },
        {
          autoResize: true,
          nodes: {
            shape: "dot",
            font: {
              face: "Inter, Segoe UI, Arial, sans-serif",
              size: largeGraph ? 10 : 12,
              color: "#1f2937",
              strokeWidth: 4,
              strokeColor: "#ffffff",
            },
          },
          groups: buildVisGroups(groups),
          edges: {
            selectionWidth: 2,
            hoverWidth: 2,
            color: { color: "#cbd5e1", highlight: "#2563eb", hover: "#2563eb" },
          },
          layout: { improvedLayout: false },
          physics: {
            stabilization: { iterations: largeGraph ? 80 : 160, updateInterval: 20 },
            barnesHut: {
              gravitationalConstant: largeGraph ? -8500 : -3600,
              centralGravity: 0.12,
              springLength: largeGraph ? 80 : 115,
              springConstant: 0.035,
              damping: 0.18,
            },
          },
          interaction: {
            hover: true,
            tooltipDelay: 120,
            hideEdgesOnDrag: true,
            navigationButtons: false,
            keyboard: false,
          },
        },
      );

      network.on("click", (params) => {
        if (!params.nodes.length) {
          setSelectedNode(null);
          return;
        }
        const node = graphData.nodes.find((candidate) => candidate.id === params.nodes[0]);
        setSelectedNode(node ?? null);
      });

      network.once("stabilizationIterationsDone", () => {
        network.setOptions({ physics: false });
      });

      networkRef.current = network;
      focusNetworkNode(network, pendingFocusRef.current, filteredGraph.nodes);
    };

    if ("requestIdleCallback" in window) {
      idleHandle = window.requestIdleCallback(renderNetwork, { timeout: 300 });
    } else {
      timerHandle = window.setTimeout(renderNetwork, 0);
    }

    return () => {
      disposed = true;
      if (idleHandle !== null && "cancelIdleCallback" in window) {
        window.cancelIdleCallback(idleHandle);
      }
      if (timerHandle !== null) {
        window.clearTimeout(timerHandle);
      }
      if (network) network.destroy();
      if (networkRef.current === network) networkRef.current = null;
    };
  }, [filteredGraph, graphData, groups, showLabels]);

  useEffect(() => {
    if (!networkRef.current || !pendingFocusId || !visibleNodeIds.has(pendingFocusId)) return;
    focusNetworkNode(networkRef.current, pendingFocusId, filteredGraph.nodes);
  }, [filteredGraph.nodes, pendingFocusId, visibleNodeIds]);

  function focusNode() {
    if (!graphData || !nodeQuery.trim()) return;
    const term = nodeQuery.trim().toLowerCase();
    const match = graphData.nodes.find(
      (node) =>
        String(node.id).toLowerCase().includes(term) ||
        String(node.label ?? "").toLowerCase().includes(term),
    );

    if (!match) {
      setSelectedNode(null);
      setPendingFocusId("");
      return;
    }

    if (!visibleNodeIds.has(match.id)) setGroupFilter("all");
    setSelectedNode(match);
    setPendingFocusId(match.id);
  }

  if (loading) {
    return <EmptyState title="Đang tải danh sách graph" text="Metadata nhỏ đang được đọc trước." />;
  }

  if (error) {
    return <EmptyState title="Không tải được danh sách graph" text={error} />;
  }

  return (
    <section className="graph-workspace">
      <aside className="graph-list" aria-label="Danh sách graph">
        {graphs.map((graph) => (
          <button
            className={graph.id === selectedGraphId ? "graph-item active" : "graph-item"}
            key={graph.id}
            onClick={() => setSelectedGraphId(graph.id)}
            onMouseEnter={() => prefetchGraph(graph)}
            type="button"
          >
            <span>{graph.title}</span>
            <strong>
              {formatNumber(graph.nodeCount)} node, {formatNumber(graph.edgeCount)} cạnh
            </strong>
          </button>
        ))}
      </aside>

      <section className="graph-panel">
        <div className="graph-header">
          <div>
            <h2>{selectedMeta?.title ?? "Graph"}</h2>
            <p>{selectedMeta?.description}</p>
          </div>
        </div>

        <div className="control-strip">
          <label>
            Group
            <select
              disabled={!graphData || groups.length === 0}
              onChange={(event) => setGroupFilter(event.target.value)}
              value={groupFilter}
            >
              <option value="all">Tất cả</option>
              {groups.map((group) => (
                <option key={String(group)} value={String(group)}>
                  {String(group)}
                </option>
              ))}
            </select>
          </label>
          <label className="check-control">
            <input
              checked={showLabels}
              onChange={(event) => setShowLabels(event.target.checked)}
              type="checkbox"
            />
            Hiện label
          </label>
          <div className="node-search">
            <input
              onChange={(event) => setNodeQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") focusNode();
              }}
              placeholder="Tìm node"
              value={nodeQuery}
            />
            <button onClick={focusNode} type="button">
              Tìm
            </button>
          </div>
        </div>

        {graphState.error && <div className="inline-message error">{graphState.error}</div>}
        {graphState.loading && <div className="inline-message">Đang tải graph JSON...</div>}

        <div className="graph-content">
          <div className="graph-canvas" ref={containerRef} />
          <aside className="node-panel">
            <span className="kicker">Node đang chọn</span>
            {selectedNode ? (
              <NodeDetails communityIndexState={communityIndexState} node={selectedNode} />
            ) : (
              <p>Chọn một node trên graph để xem thông tin nhanh.</p>
            )}
          </aside>
        </div>
      </section>
    </section>
  );
}

function NodeDetails({ communityIndexState, node }) {
  const context = getNodeContext(node, communityIndexState.data);
  const details = getNodeDetails(node, context);

  return (
    <>
      <h3>{node.label ?? node.id}</h3>
      <p>{getNodeSummary(node, context)}</p>
      {communityIndexState.loading && (
        <div className="node-context-message">Đang tải nội dung community...</div>
      )}
      {communityIndexState.error && (
        <div className="node-context-message error">{communityIndexState.error}</div>
      )}
      {context.community && (
        <section className="node-context-card">
          <span>{context.kind === "subreddit" ? "Community của subreddit" : "Nội dung community"}</span>
          <strong>{context.community.name}</strong>
          <p>{context.community.reason}</p>
          <div className="node-tag-list">
            {context.community.examples.slice(0, 10).map((example) => (
              <span key={example}>{example}</span>
            ))}
          </div>
        </section>
      )}
      <dl className="node-detail-list">
        {details.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </>
  );
}

function getNodeSummary(node, context) {
  if (context.kind === "subreddit" && context.community) {
    return `Subreddit này thuộc community "${context.community.name}". Nội dung bên dưới được suy ra từ cụm community và metadata graph, không phải toàn bộ bài viết/comment gốc.`;
  }
  if (context.kind === "community" && context.community) {
    return context.community.reason;
  }
  return stripHtml(node.title ?? node.label ?? node.id);
}

function getNodeContext(node, communityIndex) {
  if (!communityIndex) return { kind: "unknown", community: null, subreddit: getSubredditName(node) };

  const subreddit = getSubredditName(node);
  const subredditCommunityId = subreddit
    ? communityIndex.subredditToCommunity?.[subreddit]
    : undefined;
  if (subreddit && subredditCommunityId !== undefined) {
    return {
      kind: "subreddit",
      subreddit,
      community: communityIndex.communities?.[String(subredditCommunityId)] ?? null,
    };
  }

  const inferredSubreddit = getSubredditSamples(node).find(
    (sample) => communityIndex.subredditToCommunity?.[sample] !== undefined,
  );
  if (inferredSubreddit) {
    const communityId = communityIndex.subredditToCommunity[inferredSubreddit];
    return {
      kind: "community",
      subreddit: inferredSubreddit,
      community: communityIndex.communities?.[String(communityId)] ?? null,
    };
  }

  const communityId = getCommunityId(node);
  if (communityId !== null) {
    return {
      kind: "community",
      community: communityIndex.communities?.[String(communityId)] ?? null,
      subreddit: null,
    };
  }

  return { kind: "unknown", community: null, subreddit };
}

function getNodeDetails(node, context) {
  return [
    ["Loại", context.kind === "community" ? "Community" : context.kind === "subreddit" ? "Subreddit" : "Node"],
    ["Subreddit", context.subreddit ?? "-"],
    ["Community", context.community?.name ?? "-"],
    ["ID", node.id],
    ["Group", node.group ?? "-"],
    ["Size", node.size ?? node.value ?? "-"],
    ["Shape", node.shape ?? "-"],
  ].filter(([, value]) => value !== undefined && value !== "");
}

function getSubredditName(node) {
  const id = String(node.id ?? "");
  const label = String(node.label ?? "");
  const title = stripHtml(node.title ?? "");
  const titleMatch = title.match(/Subreddit:\s*([^\n]+)/i);

  if (id.startsWith("subreddit:")) return id.slice("subreddit:".length);
  if (id.startsWith("s_")) return id.slice(2);
  if (titleMatch) return titleMatch[1].trim();
  if (String(node.group ?? "") === "subreddit" && label) return label;
  if (isCommunityLikeNode(node)) return "";
  return label || id;
}

function getCommunityId(node) {
  const id = String(node.id ?? "");
  const title = stripHtml(node.title ?? "");
  const directMatch = id.match(/^community:(\d+)$/) ?? id.match(/^c_(\d+)$/);
  const titleMatch = title.match(/Community\s*:?\s*(\d+)/i) ?? title.match(/ID:\s*(\d+)/i);

  if (directMatch) return Number(directMatch[1]);
  if (titleMatch) return Number(titleMatch[1]);
  if (/^\d+$/.test(id) && /Community/i.test(title)) return Number(id);
  return null;
}

function getSubredditSamples(node) {
  const title = stripHtml(node.title ?? "");
  const topMatch = title.match(/Top subreddits:\s*([^\n]+)/i);
  if (!topMatch) return [];
  return topMatch[1]
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function isCommunityLikeNode(node) {
  const id = String(node.id ?? "");
  const title = stripHtml(node.title ?? "");
  return id.startsWith("community:") || /^c_\d+$/.test(id) || /Community/i.test(title);
}

function focusNetworkNode(network, nodeId, visibleNodes) {
  if (!nodeId || !visibleNodes.some((node) => node.id === nodeId)) return;
  window.setTimeout(() => {
    network.selectNodes([nodeId]);
    network.focus(nodeId, {
      scale: 1.35,
      animation: { duration: 420, easingFunction: "easeInOutQuad" },
    });
  }, 0);
}

function filterGraph(graphData, groupFilter) {
  if (!graphData) return { nodes: [], edges: [] };

  const nodes =
    groupFilter === "all"
      ? graphData.nodes
      : graphData.nodes.filter((node) => String(node.group ?? "none") === groupFilter);
  const allowedNodeIds = new Set(nodes.map((node) => node.id));
  const edges = graphData.edges.filter(
    (edge) => allowedNodeIds.has(edge.from) && allowedNodeIds.has(edge.to),
  );

  return { nodes, edges };
}

function normalizeEdgeColor(color) {
  if (!color) return { color: "#cbd5e1", highlight: "#2563eb", hover: "#2563eb" };
  if (typeof color === "string") {
    return { color, highlight: "#2563eb", hover: "#2563eb" };
  }
  return color;
}

function normalizeTitle(value) {
  return String(value ?? "").replace(/\n/g, "<br>");
}

function stripHtml(value) {
  return String(value ?? "")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<[^>]*>/g, "")
    .trim();
}

function buildVisGroups(groups) {
  const palette = [
    "#2563eb",
    "#059669",
    "#d97706",
    "#dc2626",
    "#7c3aed",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
    "#9333ea",
    "#0f766e",
  ];
  const result = {
    default: { color: { background: "#ffffff", border: "#94a3b8" } },
    bridge: { color: { background: "#dc2626", border: "#991b1b" } },
    gateway: { color: { background: "#059669", border: "#047857" } },
    both: { color: { background: "#2563eb", border: "#1d4ed8" } },
    other: { color: { background: "#ffffff", border: "#94a3b8" } },
    cluster: { color: { background: "#2563eb", border: "#1d4ed8" } },
    subreddit: { color: { background: "#ffffff", border: "#94a3b8" } },
  };

  groups.forEach((group, index) => {
    const color = palette[index % palette.length];
    result[String(group)] = {
      color: { background: color, border: color },
      font: { color: "#111827", strokeWidth: 5, strokeColor: "#ffffff" },
    };
  });

  return result;
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
    {
      label: "PPR seeds",
      value: data.keyMetrics.pagerankSeeds ?? 0,
      display: formatNumber(data.keyMetrics.pagerankSeeds ?? 0),
      helper: "Seed PageRank Nibble",
      maxValue: Math.max(data.keyMetrics.pagerankSeeds ?? 0, 1),
      color: chartPalette[6],
    },
    {
      label: "Best conductance",
      value: data.keyMetrics.pagerankBestConductance ?? 0,
      display: formatScore(data.keyMetrics.pagerankBestConductance ?? 0, 4),
      helper: "Cụm APPR tốt nhất",
      maxValue: 1,
      color: chartPalette[7],
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
  }));
  const topCommunities = data.topCommunities.slice(0, 12).map((community) => ({
    label: community.name,
    value: community.size,
    detail: `${formatNumber(community.size)} subreddit`,
    subtext: community.densityLabel,
  }));
  const topBridges = data.topBridges.slice(0, 12).map((row) => ({
    label: row.subreddit,
    value: row.bridge_score,
    detail: formatScore(row.bridge_score, 4),
    subtext: `${row.source_name} -> ${row.target_name}`,
  }));
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
  const pagerankSummaries = data.pagerankNibble?.summary ?? [];
  const pagerankSeeds = pagerankSummaries.slice(0, 12).map((row) => ({
    label: row.seed,
    value: 1 - Number(row.conductance ?? 0),
    detail: `φ ${formatScore(row.conductance, 4)}`,
    subtext: `${formatNumber(row.clusterSize)} node / ${row.seedCommunityName || "Community"}`,
  }));
  const pagerankTopNodes = (data.pagerankNibble?.topNodes ?? []).slice(0, 12).map((row) => ({
    label: `${row.seed} -> ${row.subreddit}`,
    value: row.pprScore,
    detail: formatScore(row.pprScore, 4),
    subtext: row.communityName || "Community",
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
  const algorithmRows = data.experiments.algorithms.map((row) => ({
    label: row.name,
    timeMinutes: parseDurationMinutes(row.time),
    detail: row.time,
    modularity: row.modularity,
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
        <HorizontalBarChart
          data={topSimilarityPairs}
          maxValue={1}
          axisFormatter={(value) => formatScore(value, 2)}
          valueFormatter={(value) => formatScore(value, 4)}
        />
      </Panel>

      <div className="stats-grid">
        <Panel eyebrow="Ranking size" title="Top community lớn">
          <HorizontalBarChart data={topCommunities} valueFormatter={formatNumber} />
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
          <HorizontalBarChart
            data={topBridges}
            axisFormatter={(value) => formatScore(value, 3)}
            valueFormatter={(value) => formatScore(value, 4)}
          />
        </Panel>
        <Panel eyebrow="Gateway role" title="Top gateway">
          <HorizontalBarChart
            data={topGateways}
            axisFormatter={(value) => formatScore(value, 2)}
            maxValue={1}
            valueFormatter={(value) => formatScore(value, 4)}
          />
        </Panel>
      </div>

      <div className="stats-grid">
        <Panel eyebrow="Path frequency" title="Top highway">
          <HorizontalBarChart data={topHighways} valueFormatter={formatNumber} />
        </Panel>
        <Panel eyebrow="Matrix" title="Highway heatmap">
          <HighwayHeatmap data={data.highwayHeatmap} />
        </Panel>
      </div>

      {pagerankSeeds.length > 0 && (
        <div className="stats-grid">
          <Panel eyebrow="Approx. PPR" title="PageRank Nibble local clusters">
            <HorizontalBarChart
              axisFormatter={(value) => formatScore(value, 2)}
              data={pagerankSeeds}
              maxValue={1}
              valueFormatter={(value) => formatScore(value, 4)}
            />
          </Panel>
          <Panel eyebrow="PPR score" title="Top node theo personalized PageRank">
            <HorizontalBarChart
              axisFormatter={(value) => formatScore(value, 2)}
              data={pagerankTopNodes}
              valueFormatter={(value) => formatScore(value, 4)}
            />
          </Panel>
        </div>
      )}

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

function HorizontalBarChart({
  data,
  maxValue,
  ranked = true,
  valueFormatter = formatNumber,
  axisFormatter = valueFormatter,
}) {
  const max = maxValue ?? Math.max(...data.map((item) => Number(item.value ?? 0)), 1);
  const ticks = buildScaleTicks(max, axisFormatter);

  return (
    <div className="hbar-chart">
      <div className={ranked ? "hbar-scale" : "hbar-scale no-rank"}>
        {ticks.map((tick) => (
          <span key={tick.ratio} style={{ left: `${tick.ratio * 100}%` }}>
            {tick.label}
          </span>
        ))}
      </div>
      {data.map((item, index) => {
        const value = Number(item.value ?? 0);
        const width = max > 0 ? clamp((value / max) * 100, value > 0 ? 2 : 0, 100) : 0;
        const color = item.color ?? chartPalette[index % chartPalette.length];

        return (
          <div
            className={ranked ? "hbar-row" : "hbar-row no-rank"}
            key={item.label}
            style={{ "--bar-color": color }}
          >
            {ranked && <div className="hbar-rank">{String(index + 1).padStart(2, "0")}</div>}
            <div className="hbar-body">
              <div className="hbar-meta">
                <span title={item.label}>{item.label}</span>
                <strong>{item.detail ?? valueFormatter(value)}</strong>
              </div>
              <div className="hbar-track">
                <div className="hbar-grid" />
                <div className="hbar-fill" style={{ width: `${width}%` }} />
              </div>
              {item.subtext && <small title={item.subtext}>{item.subtext}</small>}
            </div>
          </div>
        );
      })}
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
      <HorizontalBarChart
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
      <div className="grouped-scale">
        {ticks.map((tick) => (
          <span key={tick.ratio} style={{ left: `${tick.ratio * 100}%` }}>
            {tick.label}
          </span>
        ))}
      </div>
      {groups.map((group) => (
        <div className="grouped-row" key={group.label}>
          <div className="grouped-label">
            <strong>{group.label}</strong>
            <span>{group.note}</span>
          </div>
          <div className="grouped-bars">
            {series.map((item) => {
              const value = Number(group.values[item.key] ?? 0);
              const width = clamp((value / max) * 100, value > 0 ? 2 : 0, 100);
              return (
                <div className="grouped-bar-line" key={item.key}>
                  <span>{item.label}</span>
                  <div
                    className="grouped-track"
                    title={`${group.label} - ${item.label}: ${group.details[item.key]}`}
                  >
                    <div style={{ width: `${width}%`, backgroundColor: item.color }} />
                  </div>
                  <strong>{group.details[item.key]}</strong>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

function AlgorithmBenchmarkChart({ data, timeFormatter = formatNumber }) {
  const maxTime = Math.max(...data.map((item) => item.timeMinutes), 1);
  const minModularity = Math.min(...data.map((item) => item.modularity)) - 0.02;
  const maxModularity = Math.max(...data.map((item) => item.modularity)) + 0.02;
  const modularityRange = Math.max(maxModularity - minModularity, Number.EPSILON);
  const plot = { left: 72, right: 604, top: 28, bottom: 206 };
  const width = plot.right - plot.left;
  const height = plot.bottom - plot.top;
  const xScale = (value) =>
    plot.left + (Math.log10(Number(value ?? 0) + 1) / Math.log10(maxTime + 1)) * width;
  const yScale = (value) =>
    plot.bottom - ((Number(value ?? 0) - minModularity) / modularityRange) * height;
  const xTicks = [0, 1, 5, 15, Math.ceil(maxTime)].filter(
    (value, index, values) => value <= maxTime && values.indexOf(value) === index,
  );
  const yTicks = [minModularity, (minModularity + maxModularity) / 2, maxModularity];

  return (
    <div className="algorithm-chart">
      <svg aria-label="Benchmark thuật toán community" viewBox="0 0 640 250">
        <line className="chart-axis" x1={plot.left} x2={plot.left} y1={plot.top} y2={plot.bottom} />
        <line className="chart-axis" x1={plot.left} x2={plot.right} y1={plot.bottom} y2={plot.bottom} />
        {xTicks.map((tick) => (
          <g key={`x-${tick}`}>
            <line className="chart-grid-line" x1={xScale(tick)} x2={xScale(tick)} y1={plot.top} y2={plot.bottom} />
            <text className="chart-tick" textAnchor="middle" x={xScale(tick)} y={plot.bottom + 24}>
              {formatScore(tick, tick < 1 ? 1 : 0)}p
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
        <text className="chart-axis-label" textAnchor="middle" x={(plot.left + plot.right) / 2} y={246}>
          Thời gian xử lý (log phút)
        </text>
        <text className="chart-axis-label" textAnchor="middle" transform="rotate(-90 20 116)" x={20} y={116}>
          Modularity
        </text>
        {data.map((item, index) => {
          const color = chartPalette[index % chartPalette.length];
          return (
            <g key={item.label}>
              <circle cx={xScale(item.timeMinutes)} cy={yScale(item.modularity)} fill={color} r="9" />
              <text className="chart-point-label" x={xScale(item.timeMinutes) + 14} y={yScale(item.modularity) + 5}>
                {item.label}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="algorithm-cards">
        {data.map((item, index) => (
          <div
            className="algorithm-card"
            key={item.label}
            style={{ "--metric-color": chartPalette[index % chartPalette.length] }}
          >
            <span>{item.label}</span>
            <strong>{item.detail}</strong>
            <small>{timeFormatter(item.timeMinutes)} / Modularity {formatScore(item.modularity, 4)}</small>
          </div>
        ))}
      </div>
    </div>
  );
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

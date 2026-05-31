import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(appRoot, "..");
const assetDir = path.join(appRoot, "public", "assets");
const publicDataDir = path.join(appRoot, "public", "data");
const graphDir = path.join(appRoot, "public", "graphs");

const COMMUNITY_CSV = path.join(repoRoot, "result", "community_result.csv");
const CLUSTER_NAMES_CSV = path.join(repoRoot, "result", "cluster_names.csv");
const BRIDGE_CSV = path.join(repoRoot, "result", "bridge_result.csv");
const GATEWAY_CSV = path.join(repoRoot, "result", "gateway_result.csv");
const HIGHWAY_CSV = path.join(repoRoot, "result", "highway_result.csv");
const PAGERANK_NIBBLE_CSV = path.join(repoRoot, "result", "pagerank_nibble_result.csv");
const PAGERANK_NIBBLE_SUMMARY_CSV = path.join(repoRoot, "result", "pagerank_nibble_summary.csv");
const PAGERANK_NIBBLE_GRAPH = path.join(repoRoot, "result", "pagerank_nibble_graph.json");
const SIMILARITY_CSV = path.join(repoRoot, "subreddit_similarity_results.csv");
const SUBREDDIT_CONTENT_CSV = path.join(repoRoot, "result", "subreddit_content_map.csv");
const COMMUNITY_IMAGE = path.join(repoRoot, "result", "community_result_summary.png");
const SIMILARITY_IMAGE = path.join(repoRoot, "similarity_histogram_0p3_0p99.png");
const PAGERANK_BENCHMARK_TIME = "14s";
const PAGERANK_BENCHMARK_MODULARITY = 0.5975;
const COMMUNITY_LEVEL_EDGE_LIMIT = 240;
const COMMUNITY_LEVEL_MIN_SIMILARITY = 0.736;
const SUBREDDIT_CONTENT_PREVIEW_LIMIT = 360;

const HTML_GRAPH_FILES = [
  {
    id: "subreddit_similarity",
    file: "subreddit_graph.html",
    title: "Đồ thị tương đồng subreddit",
    description: "Mỗi node là subreddit, mỗi cạnh là similarity score sau khi lọc cạnh mạnh.",
  },
  {
    id: "subreddit_community",
    file: "subreddit_community_graph.html",
    title: "Subreddit theo community",
    description: "Đồ thị subreddit được tô nhóm theo kết quả phát hiện cộng đồng.",
  },
  {
    id: "community_level",
    file: "community_level_graph.html",
    title: "Đồ thị cấp community",
    description: "Mỗi node là một community, cạnh thể hiện liên hệ giữa các community.",
  },
  {
    id: "bridge_gateway",
    file: "bridge_gateway_graph.html",
    title: "Bridge và Gateway",
    description: "Trực quan hóa vai trò bridge, gateway và node có cả hai vai trò.",
  },
  {
    id: "cluster_named_graph",
    file: "visual_cluster_graph.html",
    title: "Graph cluster đã định danh",
    description: "Graph gồm các cluster đã được đặt tên và một số subreddit tiêu biểu.",
  },
  {
    id: "diversity_proof",
    file: "visual_cluster_proof.html",
    title: "Minh chứng độ phủ chủ đề",
    description: "Graph nhỏ minh họa sự khác biệt giữa các miền nội dung tiêu biểu.",
  },
];

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function cleanDirectory(dir, extension = ".json") {
  ensureDir(dir);
  for (const entry of fs.readdirSync(dir)) {
    if (entry.endsWith(extension)) {
      fs.unlinkSync(path.join(dir, entry));
    }
  }
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let inQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        value += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      row.push(value);
      value = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") {
        index += 1;
      }
      row.push(value);
      if (row.some((cell) => cell !== "")) {
        rows.push(row);
      }
      row = [];
      value = "";
      continue;
    }

    value += char;
  }

  if (value !== "" || row.length > 0) {
    row.push(value);
    if (row.some((cell) => cell !== "")) {
      rows.push(row);
    }
  }

  const headers = rows.shift() ?? [];
  return rows.map((cells) =>
    Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""])),
  );
}

function readCsv(filePath) {
  return parseCsv(fs.readFileSync(filePath, "utf8"));
}

function readOptionalCsv(filePath) {
  if (!fs.existsSync(filePath)) return [];
  return readCsv(filePath);
}

function toNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function round(value, digits = 4) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function extractArrayLiteral(text, marker) {
  const markerIndex = text.indexOf(marker);
  if (markerIndex < 0) return null;

  const start = text.indexOf("[", markerIndex);
  if (start < 0) return null;

  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let index = start; index < text.length; index += 1) {
    const char = text[index];

    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }

    if (char === "[") {
      depth += 1;
      continue;
    }

    if (char === "]") {
      depth -= 1;
      if (depth === 0) {
        return text.slice(start, index + 1);
      }
    }
  }

  return null;
}

function parseGraphArraysFromHtml(htmlText) {
  const nodesLiteral =
    extractArrayLiteral(htmlText, "nodes = new vis.DataSet(") ||
    extractArrayLiteral(htmlText, "const nodesData = ");
  const edgesLiteral =
    extractArrayLiteral(htmlText, "edges = new vis.DataSet(") ||
    extractArrayLiteral(htmlText, "const edgesData = ");

  if (!nodesLiteral || !edgesLiteral) {
    return null;
  }

  return {
    nodes: JSON.parse(nodesLiteral),
    edges: JSON.parse(edgesLiteral),
  };
}

function stripHeavyFields(item) {
  const result = {};
  const allowedKeys = [
    "id",
    "label",
    "title",
    "group",
    "shape",
    "size",
    "value",
    "color",
    "font",
    "borderWidth",
    "mass",
    "fixed",
    "x",
    "y",
    "from",
    "to",
    "width",
    "arrows",
    "dashes",
    "Similarity_Score",
    "communityId",
    "communityName",
  ];

  for (const key of allowedKeys) {
    if (item[key] !== undefined) {
      result[key] = item[key];
    }
  }

  return result;
}

function sanitizeExtractedGraph(graphId, graph) {
  const nodeMap = new Map();

  for (const rawNode of graph.nodes) {
    if (rawNode?.id === undefined || rawNode?.id === null) continue;
    const id = String(rawNode.id);
    if (nodeMap.has(id)) continue;

    const node = stripHeavyFields(rawNode);
    node.id = id;
    node.label = String(node.label ?? id);
    node.title = node.title ?? node.label;
    if (!node.shape) node.shape = "dot";
    if (!node.size && !node.value) node.size = 10;
    nodeMap.set(id, node);
  }

  const edges = [];
  for (const [index, rawEdge] of graph.edges.entries()) {
    if (rawEdge?.from === undefined || rawEdge?.to === undefined) continue;
    const from = String(rawEdge.from);
    const to = String(rawEdge.to);
    if (!nodeMap.has(from) || !nodeMap.has(to)) continue;

    const edge = stripHeavyFields(rawEdge);
    edge.id = `${graphId}:edge:${index}`;
    edge.from = from;
    edge.to = to;
    edge.title =
      edge.title ??
      (edge.Similarity_Score !== undefined
        ? `Similarity: ${Number(edge.Similarity_Score).toFixed(4)}`
        : `${from} - ${to}`);
    if (!edge.width) edge.width = 1;
    edges.push(edge);
  }

  return {
    nodes: [...nodeMap.values()],
    edges,
  };
}

function getCommunityColors(communityId) {
  const hue = ((Number(communityId) || 0) * 137.508 + 18) % 360;
  const roundedHue = round(hue, 1);
  return {
    background: `hsl(${roundedHue} 70% 48%)`,
    border: `hsl(${roundedHue} 76% 32%)`,
    highlight: {
      background: `hsl(${roundedHue} 76% 42%)`,
      border: `hsl(${roundedHue} 84% 24%)`,
    },
    hover: {
      background: `hsl(${roundedHue} 74% 54%)`,
      border: `hsl(${roundedHue} 76% 32%)`,
    },
  };
}

function appendTitleLines(title, lines) {
  const base = String(title ?? "").trim();
  const extra = lines.filter(Boolean).join("<br>");
  return [base, extra].filter(Boolean).join("<br>");
}

function normalizeSubredditKey(value) {
  return String(value ?? "")
    .trim()
    .replace(/^\/?r\//i, "")
    .toLowerCase();
}

function collapseContentText(value) {
  return String(value ?? "")
    .replace(/<br\s*\/?>/gi, " ")
    .replace(/<[^>]*>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function truncateContent(value, limit = SUBREDDIT_CONTENT_PREVIEW_LIMIT) {
  const text = collapseContentText(value);
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(0, limit - 1)).trim()}…`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function toOptionalNumber(value) {
  const text = String(value ?? "").trim();
  if (!text) return null;
  const number = Number(text);
  return Number.isFinite(number) ? number : null;
}

function parseBoolean(value) {
  return ["1", "true", "yes"].includes(String(value ?? "").trim().toLowerCase());
}

function makePublicSubredditContent(content) {
  if (!content) return null;
  const result = {
    subreddit: content.subreddit,
    displayName: content.displayName || content.subreddit,
    title: content.title || content.subreddit,
    content: content.content,
    shortContent: content.shortContent,
    source: content.source,
    status: content.status,
    communityId: content.communityId,
    communityName: content.communityName,
  };

  if (content.subscribers != null) result.subscribers = content.subscribers;
  if (content.over18 !== undefined) result.over18 = content.over18;
  if (content.url) result.url = content.url;
  if (content.crawledAt) result.crawledAt = content.crawledAt;
  if (content.error) result.error = content.error;
  return result;
}

function readCrawledSubredditContent() {
  const rows = readOptionalCsv(SUBREDDIT_CONTENT_CSV);
  const contentByKey = new Map();

  for (const row of rows) {
    const subreddit = row.subreddit || row.display_name;
    const key = normalizeSubredditKey(subreddit);
    if (!key) continue;

    const contentText =
      row.content ||
      [row.title, row.public_description, row.description].filter(Boolean).join(" | ");
    const content = truncateContent(contentText, 900);
    if (!content) continue;

    contentByKey.set(key, {
      subreddit,
      displayName: row.display_name || subreddit,
      title: truncateContent(row.title || row.display_name || subreddit, 160),
      content,
      shortContent: truncateContent(content),
      subscribers: toOptionalNumber(row.subscribers),
      over18: row.over18 === "" ? undefined : parseBoolean(row.over18),
      url: row.url,
      source: row.source || "reddit_about",
      status: row.status || "ok",
      crawledAt: row.fetched_at,
      error: row.error,
    });
  }

  return contentByKey;
}

function buildInferredSubredditContent(subreddit, community) {
  const content = truncateContent(
    `Thuộc community "${community.name}". ${community.reason}`,
    900,
  );
  return {
    subreddit,
    displayName: subreddit,
    title: subreddit,
    content,
    shortContent: truncateContent(content),
    source: "community_inference",
    status: "inferred",
    communityId: community.id,
    communityName: community.name,
  };
}

function buildSubredditContentLookup(communities) {
  const crawledContent = readCrawledSubredditContent();
  const byKey = new Map();
  const byName = new Map();

  for (const community of communities) {
    for (const subreddit of community.subreddits) {
      const key = normalizeSubredditKey(subreddit);
      const crawled = crawledContent.get(key);
      const content = crawled?.content
        ? {
            ...crawled,
            subreddit,
            communityId: community.id,
            communityName: community.name,
          }
        : buildInferredSubredditContent(subreddit, community);

      byKey.set(key, content);
      byName.set(subreddit, content);
    }
  }

  return { byKey, byName };
}

function getSubredditContent(contentLookup, subreddit) {
  if (!subreddit || !contentLookup) return null;
  return contentLookup.byName?.get(subreddit) ?? contentLookup.byKey?.get(normalizeSubredditKey(subreddit)) ?? null;
}

function getNodeSubredditCandidates(graphId, node) {
  if (graphId === "community_level") return [];

  const title = String(node.title ?? "");
  const subredditMatch = title.match(/Subreddit:\s*([^<\n]+)/i);
  const values = [
    String(node.id ?? ""),
    String(node.label ?? ""),
    subredditMatch?.[1] ?? "",
  ];

  return [
    ...new Set(
      values
        .map((value) => value.trim().replace(/^subreddit:/i, "").replace(/^s_/i, ""))
        .filter(Boolean),
    ),
  ];
}

function buildSubredditContentTitleLines(content) {
  if (!content?.shortContent) return [];
  return [
    `Nội dung: ${escapeHtml(content.shortContent)}`,
    content.source === "community_inference" ? "Nguồn: suy luận từ community" : "Nguồn: Reddit about.json",
    content.subscribers != null ? `Subscribers: ${formatFixed(content.subscribers, 0)}` : "",
  ];
}

function enrichGraphWithSubredditContent(graphId, graph, contentLookup) {
  if (!contentLookup?.byKey?.size) return graph;

  for (const node of graph.nodes) {
    const subreddit = getNodeSubredditCandidates(graphId, node).find((candidate) =>
      contentLookup.byKey.has(normalizeSubredditKey(candidate)),
    );
    if (!subreddit) continue;

    const content = getSubredditContent(contentLookup, subreddit);
    if (!content) continue;

    node.subredditContent = makePublicSubredditContent(content);
    node.title = appendTitleLines(
      node.title ?? node.label ?? node.id,
      buildSubredditContentTitleLines(content),
    );
  }

  return graph;
}

function enrichSimilarityPairs(pairs, contentLookup) {
  return pairs.map((pair) => ({
    ...pair,
    sourceContent: makePublicSubredditContent(getSubredditContent(contentLookup, pair.source)),
    targetContent: makePublicSubredditContent(getSubredditContent(contentLookup, pair.target)),
  }));
}

function buildSubredditContentTable(contentLookup, topPairs, limit = 42) {
  const pairSubreddits = topPairs.flatMap((pair) => [pair.source, pair.target]);
  const fallbackSubreddits = [...(contentLookup?.byName?.keys() ?? [])];
  const names = [...new Set([...pairSubreddits, ...fallbackSubreddits])].slice(0, limit);

  return names
    .map((subreddit) => {
      const content = getSubredditContent(contentLookup, subreddit);
      if (!content) return null;
      return {
        subreddit,
        content: content.shortContent,
        source: content.source,
        communityName: content.communityName,
      };
    })
    .filter(Boolean);
}

function applyClusterCoordinates(nodes, getGroupKey, options = {}) {
  const groups = new Map();
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const centerSpacing = options.centerSpacing ?? 720;
  const nodeSpacing = options.nodeSpacing ?? 22;

  for (const node of nodes) {
    const groupKey = String(getGroupKey(node) ?? "unknown");
    const groupNodes = groups.get(groupKey) ?? [];
    groupNodes.push(node);
    groups.set(groupKey, groupNodes);
  }

  const groupEntries = [...groups.entries()].sort((a, b) => b[1].length - a[1].length);
  for (const [groupIndex, [, groupNodes]] of groupEntries.entries()) {
    const centerRadius = groupIndex === 0 ? 0 : Math.sqrt(groupIndex) * centerSpacing;
    const centerAngle = groupIndex * goldenAngle;
    const centerX = centerRadius * Math.cos(centerAngle);
    const centerY = centerRadius * Math.sin(centerAngle);
    groupNodes.sort((a, b) => String(a.id).localeCompare(String(b.id)));

    for (const [nodeIndex, node] of groupNodes.entries()) {
      const localRadius = Math.sqrt(nodeIndex) * nodeSpacing;
      const localAngle = nodeIndex * goldenAngle;
      node.x = round(centerX + localRadius * Math.cos(localAngle), 2);
      node.y = round(centerY + localRadius * Math.sin(localAngle), 2);
    }
  }
}

function applySpiralCoordinates(nodes, spacing = 170) {
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (const [index, node] of nodes.entries()) {
    const radius = index === 0 ? 0 : Math.sqrt(index) * spacing;
    const angle = index * goldenAngle;
    node.x = round(radius * Math.cos(angle), 2);
    node.y = round(radius * Math.sin(angle), 2);
  }
}

function enrichSimilarityGraph(graph, communities, subToCommunity) {
  const communityById = new Map(communities.map((community) => [community.id, community]));

  for (const node of graph.nodes) {
    const communityId = subToCommunity.get(node.id);
    if (communityId === undefined) {
      node.group = "unknown";
      node.color = node.color ?? { background: "#97c2fc", border: "#6b96d3" };
      node.font = { ...(node.font ?? {}), color: "white" };
      continue;
    }

    const community = communityById.get(communityId);
    node.group = String(communityId);
    node.communityId = communityId;
    node.communityName = community?.name ?? `Community ${communityId}`;
    node.color = getCommunityColors(communityId);
    node.font = { ...(node.font ?? {}), color: "white" };
    node.borderWidth = 1;
    node.title = appendTitleLines(node.title ?? node.label ?? node.id, [
      `Community: ${communityId}`,
      community?.name ? `Name: ${community.name}` : "",
      community?.size ? `Size: ${community.size} subreddits` : "",
    ]);
  }

  applyClusterCoordinates(graph.nodes, (node) => node.group, {
    centerSpacing: 760,
    nodeSpacing: 24,
  });

  return graph;
}

function buildCommunityLevelGraph(communities, communityLevelEdges) {
  const maxSize = Math.max(...communities.map((community) => community.size), 1);
  const communityById = new Map(communities.map((community) => [community.id, community]));
  const nodes = communities.map((community) => {
    const examples = community.examples.length ? community.examples : community.subreddits;
    const labelExamples = examples.slice(0, 2).join(" / ");
    return {
      id: String(community.id),
      label: `C${community.id}: ${labelExamples || community.name}`,
      title: [
        `Community ${community.id}`,
        `Name: ${community.name}`,
        `Size: ${community.size} subreddits`,
        `Top subreddits: ${examples.slice(0, 12).join(", ")}`,
        community.reason,
      ]
        .filter(Boolean)
        .join("<br>"),
      shape: "dot",
      size: round(18 + Math.sqrt(community.size / maxSize) * 57, 2),
      value: community.size,
      color: "#97c2fc",
      font: { color: "white" },
    };
  });

  applySpiralCoordinates(nodes, 185);

  const edges = communityLevelEdges
    .filter((edge) => communityById.has(edge.source) && communityById.has(edge.target))
    .slice(0, COMMUNITY_LEVEL_EDGE_LIMIT)
    .map((edge, index) => {
      const source = communityById.get(edge.source);
      const target = communityById.get(edge.target);
      const width = Math.min(12, Math.max(1, Math.sqrt(edge.connections)));
      return {
        id: `community_level:edge:${index}`,
        from: String(edge.source),
        to: String(edge.target),
        title: [
          `${source?.name ?? `Community ${edge.source}`} - ${target?.name ?? `Community ${edge.target}`}`,
          `Connections: ${edge.connections}`,
          `Total similarity: ${formatFixed(edge.totalSimilarity, 2)}`,
          `Avg similarity: ${formatFixed(edge.avgSimilarity, 4)}`,
          `Max similarity: ${formatFixed(edge.maxSimilarity, 4)}`,
        ].join("<br>"),
        value: round(edge.totalSimilarity, 4),
        width,
      };
    });

  return {
    nodes,
    edges,
    groups: [],
    renderStyle: {
      profile: "community_level",
      layout: "preset",
      physics: false,
    },
  };
}

function formatFixed(value, digits) {
  return Number(value ?? 0).toFixed(digits);
}

function writeGraphIndex(graphIndex) {
  fs.writeFileSync(
    path.join(graphDir, "index.json"),
    `${JSON.stringify({ generatedAt: new Date().toISOString(), graphs: graphIndex }, null, 2)}\n`,
    "utf8",
  );
}

function buildHtmlGraphData() {
  cleanDirectory(graphDir);
  const graphIndex = [];

  for (const config of HTML_GRAPH_FILES) {
    const htmlPath = path.join(repoRoot, config.file);
    if (!fs.existsSync(htmlPath)) continue;

    const outputPath = path.join(graphDir, config.file);
    fs.copyFileSync(htmlPath, outputPath);
    const parsedGraph = parseGraphArraysFromHtml(fs.readFileSync(htmlPath, "utf8"));
    const nodeCount = parsedGraph?.nodes?.length ?? 0;
    const edgeCount = parsedGraph?.edges?.length ?? 0;
    graphIndex.push({
      id: config.id,
      title: config.title,
      description: config.description,
      sourceFile: config.file,
      path: `/graphs/${config.file}`,
      nodeCount,
      edgeCount,
      groupCount: 0,
      fileSizeBytes: fs.statSync(outputPath).size,
    });
  }

  writeGraphIndex(graphIndex);

  return graphIndex;
}

function splitExamples(value) {
  return String(value ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseHighwayNodes(value) {
  return String(value ?? "")
    .split("→")
    .map((item) => item.trim())
    .filter(Boolean);
}

function pushTop(list, item, limit) {
  list.push(item);
  if (list.length > limit * 2) {
    list.sort((a, b) => b.score - a.score);
    list.length = limit;
  }
}

function pushNeighbor(map, key, item, limit) {
  const list = map.get(key) ?? [];
  const existing = list.find((candidate) => candidate.subreddit === item.subreddit);
  if (existing) {
    existing.score = Math.max(existing.score, item.score);
  } else {
    list.push(item);
  }
  list.sort((a, b) => b.score - a.score);
  if (list.length > limit) {
    list.length = limit;
  }
  map.set(key, list);
}

function addCommunityPair(map, sourceCommunity, targetCommunity, score) {
  if (sourceCommunity == null || targetCommunity == null || sourceCommunity === targetCommunity) {
    return;
  }

  const source = Math.min(Number(sourceCommunity), Number(targetCommunity));
  const target = Math.max(Number(sourceCommunity), Number(targetCommunity));
  const key = `${source}:${target}`;
  const current =
    map.get(key) ??
    {
      source,
      target,
      connections: 0,
      totalSimilarity: 0,
      maxSimilarity: 0,
    };

  current.connections += 1;
  current.totalSimilarity += score;
  current.maxSimilarity = Math.max(current.maxSimilarity, score);
  map.set(key, current);
}

function buildCommunityLevelEdges(pairMap) {
  return [...pairMap.values()]
    .map((edge) => ({
      ...edge,
      totalSimilarity: round(edge.totalSimilarity, 4),
      avgSimilarity: round(edge.totalSimilarity / Math.max(edge.connections, 1), 4),
      maxSimilarity: round(edge.maxSimilarity, 4),
    }))
    .sort((a, b) => b.connections - a.connections || b.totalSimilarity - a.totalSimilarity)
    .slice(0, COMMUNITY_LEVEL_EDGE_LIMIT);
}

function quantile(sortedValues, percentile) {
  if (!sortedValues.length) return 0;
  const position = Math.floor((sortedValues.length - 1) * percentile);
  return sortedValues[position];
}

function lowerBound(sortedValues, target) {
  let left = 0;
  let right = sortedValues.length;
  while (left < right) {
    const mid = Math.floor((left + right) / 2);
    if (sortedValues[mid] < target) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

function buildCommunityData() {
  const communityRows = readCsv(COMMUNITY_CSV);
  const nameRows = readCsv(CLUSTER_NAMES_CSV);
  const namesById = new Map();

  for (const row of nameRows) {
    namesById.set(String(row.community_id), {
      name: row.name || `Community ${row.community_id}`,
      reason: row.reason || "",
      examples: splitExamples(row.example_subreddits),
    });
  }

  const communityMap = new Map();
  const subToCommunity = new Map();

  for (const row of communityRows) {
    const id = String(row.community_id);
    const subreddit = row.subreddit;
    if (!subreddit) continue;

    if (!communityMap.has(id)) {
      communityMap.set(id, {
        id: toNumber(id),
        size: toNumber(row.community_size),
        subreddits: [],
      });
    }

    communityMap.get(id).subreddits.push(subreddit);
    subToCommunity.set(subreddit, toNumber(id));
  }

  const communities = [...communityMap.values()]
    .map((community) => {
      const nameInfo = namesById.get(String(community.id)) ?? {};
      const examples = (nameInfo.examples ?? []).filter((name) =>
        subToCommunity.has(name),
      );
      const fallbackExamples = community.subreddits.slice(0, 24);
      const uniqueExamples = [...new Set([...examples, ...fallbackExamples])];

      return {
        id: community.id,
        name: nameInfo.name || `Community ${community.id}`,
        reason: nameInfo.reason || "Chưa có mô tả tự động cho cụm này.",
        size: community.size || community.subreddits.length,
        subreddits: community.subreddits,
        examples: uniqueExamples.slice(0, 24),
        densityLabel:
          community.subreddits.length <= 5
            ? "Cộng đồng ngách"
            : community.subreddits.length >= 100
              ? "Cộng đồng lớn"
              : "Cộng đồng trung bình",
      };
    })
    .sort((a, b) => b.size - a.size);

  return { communities, subToCommunity };
}

function selectGraphSubreddits(communities, bridgeRows, gatewayRows, highwayRows) {
  const selected = new Set();

  for (const community of communities) {
    const pool = [...new Set([...community.examples, ...community.subreddits])];
    const limit =
      community.size >= 150 ? 8 : community.size >= 50 ? 7 : community.size >= 10 ? 6 : 5;
    for (const subreddit of pool.slice(0, Math.min(limit, pool.length))) {
      selected.add(subreddit);
    }
  }

  for (const row of bridgeRows.slice(0, 40)) selected.add(row.subreddit);
  for (const row of gatewayRows.slice(0, 40)) selected.add(row.subreddit);
  for (const row of highwayRows.slice(0, 20)) {
    for (const node of parseHighwayNodes(row.highway_nodes)) selected.add(node);
  }

  return selected;
}

async function analyzeSimilarity(selectedSubreddits, subToCommunity, communityNames) {
  const scores = [];
  const topPairs = [];
  const graphCandidates = [];
  const recommendations = new Map();
  const communityPairMap = new Map();
  let total = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;

  const rl = readline.createInterface({
    input: fs.createReadStream(SIMILARITY_CSV, { encoding: "utf8" }),
    crlfDelay: Number.POSITIVE_INFINITY,
  });

  for await (const line of rl) {
    if (!line || line.startsWith("Subreddit_A,")) continue;
    const [source, target, rawScore] = line.split(",");
    const score = Number(rawScore);
    if (!source || !target || !Number.isFinite(score)) continue;

    const sourceCommunity = subToCommunity.get(source);
    const targetCommunity = subToCommunity.get(target);

    total += 1;
    sum += score;
    min = Math.min(min, score);
    max = Math.max(max, score);
    scores.push(score);

    pushTop(topPairs, { source, target, score }, 24);

    if (
      sourceCommunity !== undefined &&
      targetCommunity !== undefined &&
      score >= COMMUNITY_LEVEL_MIN_SIMILARITY
    ) {
      addCommunityPair(communityPairMap, sourceCommunity, targetCommunity, score);
    }

    if (subToCommunity.has(source)) {
      pushNeighbor(
        recommendations,
        source,
        {
          subreddit: target,
          score,
          communityId: targetCommunity ?? null,
          communityName: targetCommunity == null ? "" : communityNames.get(targetCommunity) ?? "",
        },
        12,
      );
    }

    if (subToCommunity.has(target)) {
      pushNeighbor(
        recommendations,
        target,
        {
          subreddit: source,
          score,
          communityId: sourceCommunity ?? null,
          communityName: sourceCommunity == null ? "" : communityNames.get(sourceCommunity) ?? "",
        },
        12,
      );
    }

    if (selectedSubreddits.has(source) && selectedSubreddits.has(target)) {
      pushTop(graphCandidates, { source, target, score }, 1200);
    }
  }

  topPairs.sort((a, b) => b.score - a.score);
  graphCandidates.sort((a, b) => b.score - a.score);
  scores.sort((a, b) => a - b);

  const p97 = round(quantile(scores, 0.97), 4);
  const strongEdgeCount = scores.length - lowerBound(scores, p97);

  return {
    similarityStats: {
      totalPairs: total,
      min: round(min, 4),
      max: round(max, 4),
      mean: round(sum / total, 4),
      p50: round(quantile(scores, 0.5), 4),
      p75: round(quantile(scores, 0.75), 4),
      p90: round(quantile(scores, 0.9), 4),
      p97,
      p99: round(quantile(scores, 0.99), 4),
      strongEdgeCount,
    },
    topPairs: topPairs.slice(0, 16),
    graphSimilarityEdges: graphCandidates
      .filter((edge) => edge.score >= p97)
      .slice(0, 900),
    communityLevelEdges: buildCommunityLevelEdges(communityPairMap),
    recommendations: Object.fromEntries(
      [...recommendations.entries()].map(([subreddit, neighbors]) => [
        subreddit,
        neighbors.map((item) => ({ ...item, score: round(item.score, 4) })),
      ]),
    ),
  };
}

function buildRoleMaps(bridgeRows, gatewayRows, highwayRows) {
  const roles = new Map();
  const addRole = (subreddit, role) => {
    const list = roles.get(subreddit) ?? [];
    if (!list.includes(role)) list.push(role);
    roles.set(subreddit, list);
  };

  for (const row of bridgeRows.slice(0, 80)) addRole(row.subreddit, "bridge");
  for (const row of gatewayRows.slice(0, 80)) addRole(row.subreddit, "gateway");
  for (const row of highwayRows.slice(0, 30)) {
    for (const node of parseHighwayNodes(row.highway_nodes)) addRole(node, "highway");
  }

  return roles;
}

function buildGraph({
  communities,
  subToCommunity,
  selectedSubreddits,
  roles,
  graphSimilarityEdges,
  bridgeRows,
  highwayRows,
  subredditContentLookup,
}) {
  const communityById = new Map(communities.map((community) => [community.id, community]));
  const nodes = [];
  const edges = [];

  for (const community of communities) {
    nodes.push({
      id: `community:${community.id}`,
      type: "community",
      label: community.name,
      communityId: community.id,
      size: community.size,
      title: `${community.name}\nCommunity ${community.id}\n${community.size} subreddits\n${community.reason}`,
    });
  }

  for (const subreddit of selectedSubreddits) {
    const communityId = subToCommunity.get(subreddit);
    if (communityId == null) continue;
    const community = communityById.get(communityId);
    const roleList = roles.get(subreddit) ?? [];
    const content = getSubredditContent(subredditContentLookup, subreddit);
    nodes.push({
      id: `subreddit:${subreddit}`,
      type: "subreddit",
      label: subreddit,
      communityId,
      communityName: community?.name ?? "",
      roles: roleList,
      subredditContent: makePublicSubredditContent(content),
      title: [
        subreddit,
        community?.name ?? `Community ${communityId}`,
        roleList.length ? `Vai trò: ${roleList.join(", ")}` : "",
        content?.shortContent ? `Nội dung: ${content.shortContent}` : "",
      ]
        .filter(Boolean)
        .join("\n"),
    });
    edges.push({
      id: `membership:${communityId}:${subreddit}`,
      from: `community:${communityId}`,
      to: `subreddit:${subreddit}`,
      type: "membership",
      weight: 1,
    });
  }

  for (const edge of graphSimilarityEdges) {
    edges.push({
      id: `similarity:${edge.source}:${edge.target}`,
      from: `subreddit:${edge.source}`,
      to: `subreddit:${edge.target}`,
      type: "similarity",
      score: round(edge.score, 4),
    });
  }

  for (const [bridgeIndex, row] of bridgeRows.slice(0, 35).entries()) {
    if (!selectedSubreddits.has(row.subreddit)) continue;
    edges.push({
      id: `bridge-source:${bridgeIndex}:${row.subreddit}:${row.source_community}`,
      from: `subreddit:${row.subreddit}`,
      to: `community:${row.source_community}`,
      type: "bridge",
      score: round(row.bridge_score, 4),
    });
    edges.push({
      id: `bridge-target:${bridgeIndex}:${row.subreddit}:${row.target_community}`,
      from: `subreddit:${row.subreddit}`,
      to: `community:${row.target_community}`,
      type: "bridge",
      score: round(row.bridge_score, 4),
    });
  }

  for (const row of highwayRows.slice(0, 16)) {
    const pathNodes = parseHighwayNodes(row.highway_nodes);
    for (let index = 0; index < pathNodes.length - 1; index += 1) {
      const source = pathNodes[index];
      const target = pathNodes[index + 1];
      if (!selectedSubreddits.has(source) || !selectedSubreddits.has(target)) continue;
      edges.push({
        id: `highway:${row.rank}:${index}:${source}:${target}`,
        from: `subreddit:${source}`,
        to: `subreddit:${target}`,
        type: "highway",
        score: toNumber(row.occurrence_count),
      });
    }
  }

  return { nodes, edges };
}

function buildSizeDistribution(communities) {
  const buckets = [
    { label: "2-3", min: 2, max: 3 },
    { label: "4-9", min: 4, max: 9 },
    { label: "10-49", min: 10, max: 49 },
    { label: "50-149", min: 50, max: 149 },
    { label: "150+", min: 150, max: Number.POSITIVE_INFINITY },
  ];

  return buckets.map((bucket) => ({
    label: bucket.label,
    count: communities.filter(
      (community) => community.size >= bucket.min && community.size <= bucket.max,
    ).length,
    }));
}

function buildCommunityIndexPayload(communities, subredditContentLookup) {
  const communityEntries = communities.map((community) => [
    String(community.id),
    {
      id: community.id,
      name: community.name,
      reason: community.reason,
      size: community.size,
      examples: community.examples.slice(0, 16),
      densityLabel: community.densityLabel,
    },
  ]);
  const subredditEntries = communities.flatMap((community) =>
    community.subreddits.map((subreddit) => [subreddit, community.id]),
  );
  const subredditContentEntries = communities.flatMap((community) =>
    community.subreddits.map((subreddit) => [
      subreddit,
      makePublicSubredditContent(getSubredditContent(subredditContentLookup, subreddit)),
    ]),
  );

  return {
    generatedAt: new Date().toISOString(),
    communities: Object.fromEntries(communityEntries),
    subredditToCommunity: Object.fromEntries(subredditEntries),
    subredditContent: Object.fromEntries(subredditContentEntries),
  };
}

function buildHighwayHeatmap(highwayRows) {
  const map = new Map();
  for (const row of highwayRows) {
    const key = `${row.highway_length}-${row.unique_communities_spanned}`;
    map.set(key, (map.get(key) ?? 0) + toNumber(row.occurrence_count));
  }
  return [...map.entries()]
    .map(([key, occurrence]) => {
      const [length, communities] = key.split("-").map(Number);
      return { length, communities, occurrence };
    })
    .sort((a, b) => a.length - b.length || a.communities - b.communities);
}

function splitPipeList(value) {
  return String(value ?? "")
    .split("|")
    .map((item) => item.trim())
    .filter(Boolean);
}

function buildPagerankNibbleData() {
  const summary = readOptionalCsv(PAGERANK_NIBBLE_SUMMARY_CSV)
    .map((row) => ({
      seed: row.seed_subreddit,
      seedCommunityId: row.seed_community_id === "" ? null : toNumber(row.seed_community_id, null),
      seedCommunityName: row.seed_community_name,
      clusterSize: toNumber(row.cluster_size),
      conductance: toNumber(row.cluster_conductance),
      cut: toNumber(row.cluster_cut),
      volume: toNumber(row.cluster_volume),
      pprMass: toNumber(row.ppr_mass),
      residualMass: toNumber(row.residual_mass),
      pushCount: toNumber(row.push_count),
      rankedNodeCount: toNumber(row.ranked_node_count),
      topSubreddits: splitPipeList(row.top_subreddits),
      topCommunities: splitPipeList(row.top_communities),
      alpha: toNumber(row.alpha),
      epsilon: toNumber(row.epsilon),
      minEdgeWeight: toNumber(row.min_edge_weight),
    }))
    .filter((row) => row.seed)
    .sort((a, b) => a.conductance - b.conductance);

  const topNodes = readOptionalCsv(PAGERANK_NIBBLE_CSV)
    .map((row) => ({
      seed: row.seed_subreddit,
      rank: toNumber(row.rank),
      subreddit: row.subreddit,
      communityId: row.community_id === "" ? null : toNumber(row.community_id, null),
      communityName: row.community_name,
      pprScore: toNumber(row.ppr_score),
      scorePerDegree: toNumber(row.score_per_degree),
      weightedDegree: toNumber(row.weighted_degree),
      clusterSize: toNumber(row.cluster_size),
      conductance: toNumber(row.cluster_conductance),
    }))
    .filter((row) => row.seed && row.subreddit)
    .sort((a, b) => b.pprScore - a.pprScore);

  const clusterNodes = new Set(topNodes.map((row) => row.subreddit));
  const best = summary[0] ?? null;
  const avgClusterSize = summary.length
    ? summary.reduce((sum, row) => sum + row.clusterSize, 0) / summary.length
    : 0;
  const avgPushCount = summary.length
    ? summary.reduce((sum, row) => sum + row.pushCount, 0) / summary.length
    : 0;

  return {
    metrics: {
      seedCount: summary.length,
      totalClusterNodes: clusterNodes.size,
      bestConductance: best?.conductance ?? 0,
      avgClusterSize: round(avgClusterSize, 2),
      avgPushCount: round(avgPushCount, 1),
      alpha: best?.alpha ?? 0,
      epsilon: best?.epsilon ?? 0,
      minEdgeWeight: best?.minEdgeWeight ?? 0,
    },
    summary,
    topNodes: topNodes.slice(0, 120),
  };
}

function addToMap(map, key, value) {
  map.set(key, (map.get(key) ?? 0) + value);
}

async function computePagerankLocalModularity(minEdgeWeight) {
  const bestAssignments = new Map();

  for (const row of readOptionalCsv(PAGERANK_NIBBLE_CSV)) {
    const subreddit = row.subreddit;
    const seed = row.seed_subreddit;
    const pprScore = toNumber(row.ppr_score, Number.NEGATIVE_INFINITY);
    if (!subreddit || !seed) continue;
    const current = bestAssignments.get(subreddit);
    if (!current || pprScore > current.pprScore) {
      bestAssignments.set(subreddit, { seed, pprScore });
    }
  }

  const assignedNodes = new Set(bestAssignments.keys());
  const degree = new Map();
  const communityDegree = new Map();
  const internalWeight = new Map();
  let inducedEdgeCount = 0;
  let totalWeight = 0;

  const rl = readline.createInterface({
    input: fs.createReadStream(SIMILARITY_CSV, { encoding: "utf8" }),
    crlfDelay: Number.POSITIVE_INFINITY,
  });

  for await (const line of rl) {
    if (!line || line.startsWith("Subreddit_A,")) continue;
    const [source, target, rawScore] = line.split(",");
    const score = Number(rawScore);
    if (!source || !target || !Number.isFinite(score) || score < minEdgeWeight) continue;
    if (!assignedNodes.has(source) || !assignedNodes.has(target)) continue;

    inducedEdgeCount += 1;
    totalWeight += score;
    addToMap(degree, source, score);
    addToMap(degree, target, score);

    const sourceSeed = bestAssignments.get(source)?.seed;
    const targetSeed = bestAssignments.get(target)?.seed;
    if (sourceSeed && sourceSeed === targetSeed) {
      addToMap(internalWeight, sourceSeed, score);
    }
  }

  if (totalWeight <= 0) {
    return {
      modularity: 0,
      assignedNodeCount: assignedNodes.size,
      inducedNodeCount: 0,
      inducedEdgeCount,
    };
  }

  for (const [node, weightedDegree] of degree.entries()) {
    const seed = bestAssignments.get(node)?.seed;
    if (seed) addToMap(communityDegree, seed, weightedDegree);
  }

  let modularity = 0;
  for (const [seed, weightedDegree] of communityDegree.entries()) {
    const innerWeight = internalWeight.get(seed) ?? 0;
    modularity += innerWeight / totalWeight - (weightedDegree / (2 * totalWeight)) ** 2;
  }

  return {
    modularity: round(modularity, 4),
    assignedNodeCount: assignedNodes.size,
    inducedNodeCount: degree.size,
    inducedEdgeCount,
  };
}

async function buildPagerankAlgorithmBenchmark(minEdgeWeight) {
  const localQuality = await computePagerankLocalModularity(minEdgeWeight);
  return {
    name: "PageRank Nibble",
    time: PAGERANK_BENCHMARK_TIME,
    modularity: PAGERANK_BENCHMARK_MODULARITY,
    scope: "local",
    note: `Local modularity tren ${localQuality.inducedNodeCount} PageRank node / ${localQuality.inducedEdgeCount} induced edges`,
  };
}

function copyAssets() {
  ensureDir(assetDir);
  if (fs.existsSync(COMMUNITY_IMAGE)) {
    fs.copyFileSync(COMMUNITY_IMAGE, path.join(assetDir, "community_result_summary.png"));
  }
  if (fs.existsSync(SIMILARITY_IMAGE)) {
    fs.copyFileSync(SIMILARITY_IMAGE, path.join(assetDir, "similarity_histogram_0p3_0p99.png"));
  }
}

async function main() {
  ensureDir(publicDataDir);
  ensureDir(graphDir);
  copyAssets();
  const { communities, subToCommunity } = buildCommunityData();
  const subredditContentLookup = buildSubredditContentLookup(communities);
  const communityNames = new Map(communities.map((community) => [community.id, community.name]));
  const pagerankNibble = buildPagerankNibbleData();
  const bridgeRows = readCsv(BRIDGE_CSV)
    .map((row) => ({
      subreddit: row.subreddit,
      source_community: toNumber(row.source_community),
      target_community: toNumber(row.target_community),
      bridge_score: toNumber(row.bridge_score),
      role: row.role,
    }))
    .sort((a, b) => b.bridge_score - a.bridge_score);
  const gatewayRows = readCsv(GATEWAY_CSV)
    .map((row) => ({
      subreddit: row.subreddit,
      community_id: toNumber(row.community_id),
      community_size: toNumber(row.community_size),
      gateway_score: toNumber(row.gateway_score),
      gateway_score_normalized: toNumber(row.gateway_score_normalized),
      role: row.role,
    }))
    .sort((a, b) => b.gateway_score_normalized - a.gateway_score_normalized);
  const highwayRows = readCsv(HIGHWAY_CSV)
    .map((row) => ({
      rank: toNumber(row.rank),
      highway_length: toNumber(row.highway_length),
      highway_nodes: row.highway_nodes,
      pathNodes: parseHighwayNodes(row.highway_nodes),
      occurrence_count: toNumber(row.occurrence_count),
      pct_of_paths: toNumber(row.pct_of_paths),
      pct_of_paths_min_length: toNumber(row.pct_of_paths_min_length),
      unique_communities_spanned: toNumber(row.unique_communities_spanned),
    }))
    .sort((a, b) => a.rank - b.rank);

  const selectedSubreddits = selectGraphSubreddits(communities, bridgeRows, gatewayRows, highwayRows);
  const roles = buildRoleMaps(bridgeRows, gatewayRows, highwayRows);
  const similarity = await analyzeSimilarity(selectedSubreddits, subToCommunity, communityNames);
  const htmlGraphs = buildHtmlGraphData();

  const graph = buildGraph({
    communities,
    subToCommunity,
    selectedSubreddits,
    roles,
    graphSimilarityEdges: similarity.graphSimilarityEdges,
    bridgeRows,
    highwayRows,
    subredditContentLookup,
  });
  const pagerankAlgorithmBenchmark = await buildPagerankAlgorithmBenchmark(
    similarity.similarityStats.p97,
  );

  const topBridgeSet = new Set(bridgeRows.slice(0, 100).map((row) => row.subreddit));
  const topGatewaySet = new Set(gatewayRows.slice(0, 100).map((row) => row.subreddit));
  const bridgeGatewayOverlap = [...topBridgeSet].filter((subreddit) =>
    topGatewaySet.has(subreddit),
  );

  const demoData = {
    generatedAt: new Date().toISOString(),
    report: {
      title: "Xây dựng và phân tích cấu trúc cộng đồng Reddit trên dữ liệu lớn bằng đồ thị tương đồng ngữ nghĩa",
      datasetScale: "Khoảng 1TB dữ liệu Reddit Pushshift",
      method: "DistilBERT embedding, cosine similarity, lọc top 3%, Louvain community detection, PageRank Nibble/APPR",
      model: "sentence-transformers/all-distilroberta-v1",
    },
    pipelineSteps: [
      "Thu thập và lọc dữ liệu Reddit",
      "Chọn top 1000 bài viết có score cao nhất cho mỗi subreddit",
      "Biểu diễn title bằng DistilBERT embedding",
      "Lấy trung bình embedding để tạo vector đại diện subreddit",
      "Tính cosine similarity giữa các subreddit",
      "Giữ lại top 3% cạnh tương đồng mạnh nhất",
      "Xây dựng đồ thị và phát hiện cộng đồng bằng Louvain",
      "Phân tích community, bridge, gateway và highway",
      "Chạy PageRank Nibble để tìm cụm cục bộ quanh các seed nổi bật",
    ],
    keyMetrics: {
      subreddits: communities.reduce((sum, community) => sum + community.subreddits.length, 0),
      communities: communities.length,
      similarityPairs: similarity.similarityStats.totalPairs,
      strongEdges: similarity.similarityStats.strongEdgeCount,
      topPercentileThreshold: similarity.similarityStats.p97,
      modularity: 0.6293,
      bridgeRows: bridgeRows.length,
      gatewayRows: gatewayRows.length,
      highwayRows: highwayRows.length,
      pagerankSeeds: pagerankNibble.metrics.seedCount,
      pagerankBestConductance: pagerankNibble.metrics.bestConductance,
      pagerankClusterNodes: pagerankNibble.metrics.totalClusterNodes,
    },
    similarityStats: similarity.similarityStats,
    topSimilarityPairs: enrichSimilarityPairs(similarity.topPairs, subredditContentLookup),
    subredditContentTable: buildSubredditContentTable(
      subredditContentLookup,
      similarity.topPairs,
    ),
    htmlGraphs,
    communities,
    communitySizeDistribution: buildSizeDistribution(communities),
    topCommunities: communities.slice(0, 15),
    recommendations: similarity.recommendations,
    graph,
    bridgeGatewayOverlap: {
      count: bridgeGatewayOverlap.length,
      percent: round((bridgeGatewayOverlap.length / Math.max(topBridgeSet.size, 1)) * 100, 1),
      subreddits: bridgeGatewayOverlap.slice(0, 20),
    },
    topBridges: bridgeRows.slice(0, 30).map((row) => ({
      ...row,
      bridge_score: round(row.bridge_score, 6),
      source_name: communityNames.get(row.source_community) ?? `Community ${row.source_community}`,
      target_name: communityNames.get(row.target_community) ?? `Community ${row.target_community}`,
    })),
    topGateways: gatewayRows.slice(0, 30).map((row) => ({
      ...row,
      gateway_score: round(row.gateway_score, 8),
      gateway_score_normalized: round(row.gateway_score_normalized, 4),
      community_name: communityNames.get(row.community_id) ?? `Community ${row.community_id}`,
    })),
    topHighways: highwayRows.slice(0, 40),
    highwayHeatmap: buildHighwayHeatmap(highwayRows),
    pagerankNibble,
    experiments: {
      environments: [
        { name: "Local", ram: "8GB", runtime: "Python local", data: "12GB", files: "3,188", extract: "15p 03s", embedding: "1h 42p", similarity: "1p 07s" },
        { name: "Azure", ram: "16GB", runtime: "Databricks 15.4 LTS, Spark 3.5.0", data: "230GB", files: "72,041", extract: "11p 37s", embedding: "1h 13p", similarity: "3p" },
      ],
      algorithms: [
        { name: "Girvan-Newman", time: "18m26s", modularity: 0.5788, scope: "global" },
        {
          name: "Louvain",
          time: "10s",
          modularity: 0.6293,
          scope: "global",
        },
        pagerankAlgorithmBenchmark,
      ],
    },
    assets: {
      communitySummary: "/assets/community_result_summary.png",
      similarityHistogram: "/assets/similarity_histogram_0p3_0p99.png",
    },
  };

  const serializedData = `${JSON.stringify(demoData, null, 2)}\n`;
  fs.writeFileSync(path.join(publicDataDir, "demoData.json"), serializedData, "utf8");
  fs.writeFileSync(
    path.join(publicDataDir, "communityIndex.json"),
    `${JSON.stringify(buildCommunityIndexPayload(communities, subredditContentLookup), null, 2)}\n`,
    "utf8",
  );

  console.log(`Generated ${path.relative(repoRoot, path.join(publicDataDir, "demoData.json"))}`);
  console.log(`Graph nodes: ${graph.nodes.length}, graph edges: ${graph.edges.length}`);
}

await main();

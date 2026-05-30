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
const SIMILARITY_CSV = path.join(repoRoot, "subreddit_similarity_results.csv");
const COMMUNITY_IMAGE = path.join(repoRoot, "result", "community_result_summary.png");
const SIMILARITY_IMAGE = path.join(repoRoot, "similarity_histogram_0p3_0p99.png");

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
    "x",
    "y",
    "from",
    "to",
    "width",
    "arrows",
    "dashes",
    "Similarity_Score",
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

function buildHtmlGraphData() {
  cleanDirectory(graphDir);
  const graphIndex = [];

  for (const config of HTML_GRAPH_FILES) {
    const htmlPath = path.join(repoRoot, config.file);
    if (!fs.existsSync(htmlPath)) continue;

    const htmlText = fs.readFileSync(htmlPath, "utf8");
    const parsedGraph = parseGraphArraysFromHtml(htmlText);
    if (!parsedGraph) continue;

    const graph = sanitizeExtractedGraph(config.id, parsedGraph);
    const groups = [...new Set(graph.nodes.map((node) => node.group).filter(Boolean))].sort();
    const graphPayload = {
      id: config.id,
      title: config.title,
      description: config.description,
      sourceFile: config.file,
      extractedFromHtml: true,
      nodes: graph.nodes,
      edges: graph.edges,
      groups,
    };

    const outputPath = path.join(graphDir, `${config.id}.json`);
    fs.writeFileSync(outputPath, `${JSON.stringify(graphPayload)}\n`, "utf8");

    graphIndex.push({
      id: config.id,
      title: config.title,
      description: config.description,
      sourceFile: config.file,
      path: `/graphs/${config.id}.json`,
      nodeCount: graph.nodes.length,
      edgeCount: graph.edges.length,
      groupCount: groups.length,
      fileSizeBytes: fs.statSync(outputPath).size,
    });
  }

  fs.writeFileSync(
    path.join(graphDir, "index.json"),
    `${JSON.stringify({ generatedAt: new Date().toISOString(), graphs: graphIndex }, null, 2)}\n`,
    "utf8",
  );

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

    total += 1;
    sum += score;
    min = Math.min(min, score);
    max = Math.max(max, score);
    scores.push(score);

    pushTop(topPairs, { source, target, score }, 24);

    if (subToCommunity.has(source)) {
      const targetCommunity = subToCommunity.get(target);
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
      const sourceCommunity = subToCommunity.get(source);
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

function buildGraph({ communities, subToCommunity, selectedSubreddits, roles, graphSimilarityEdges, bridgeRows, highwayRows }) {
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
    nodes.push({
      id: `subreddit:${subreddit}`,
      type: "subreddit",
      label: subreddit,
      communityId,
      communityName: community?.name ?? "",
      roles: roleList,
      title: `${subreddit}\n${community?.name ?? `Community ${communityId}`}${roleList.length ? `\nVai trò: ${roleList.join(", ")}` : ""}`,
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

function buildCommunityIndexPayload(communities) {
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

  return {
    generatedAt: new Date().toISOString(),
    communities: Object.fromEntries(communityEntries),
    subredditToCommunity: Object.fromEntries(subredditEntries),
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
  const htmlGraphs = buildHtmlGraphData();

  const { communities, subToCommunity } = buildCommunityData();
  const communityNames = new Map(communities.map((community) => [community.id, community.name]));
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
  const graph = buildGraph({
    communities,
    subToCommunity,
    selectedSubreddits,
    roles,
    graphSimilarityEdges: similarity.graphSimilarityEdges,
    bridgeRows,
    highwayRows,
  });

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
      method: "DistilBERT embedding, cosine similarity, lọc top 3%, Louvain community detection",
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
    },
    similarityStats: similarity.similarityStats,
    topSimilarityPairs: similarity.topPairs,
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
    experiments: {
      environments: [
        { name: "Local", ram: "8GB", runtime: "Python local", data: "12GB", files: "3,188", extract: "15p 03s", embedding: "1h 42p", similarity: "1p 07s" },
        { name: "Azure", ram: "16GB", runtime: "Databricks 15.4 LTS, Spark 3.5.0", data: "230GB", files: "72,041", extract: "11p 37s", embedding: "1h 13p", similarity: "3p" },
      ],
      algorithms: [
        { name: "Girvan-Newman", time: "18m26s", modularity: 0.5788 },
        { name: "Louvain", time: "10s", modularity: 0.6293 },
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
    `${JSON.stringify(buildCommunityIndexPayload(communities), null, 2)}\n`,
    "utf8",
  );

  console.log(`Generated ${path.relative(repoRoot, path.join(publicDataDir, "demoData.json"))}`);
  console.log(`Graph nodes: ${graph.nodes.length}, graph edges: ${graph.edges.length}`);
}

await main();

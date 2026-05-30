from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_CSV = REPO_ROOT / "subreddit_similarity_results.csv"
DEFAULT_COMMUNITY_CSV = REPO_ROOT / "result" / "community_result.csv"
DEFAULT_CLUSTER_NAMES_CSV = REPO_ROOT / "result" / "cluster_names.csv"
DEFAULT_BRIDGE_CSV = REPO_ROOT / "result" / "bridge_result.csv"
DEFAULT_GATEWAY_CSV = REPO_ROOT / "result" / "gateway_result.csv"
DEFAULT_HIGHWAY_CSV = REPO_ROOT / "result" / "highway_result.csv"
DEFAULT_RESULT_CSV = REPO_ROOT / "result" / "pagerank_nibble_result.csv"
DEFAULT_SUMMARY_CSV = REPO_ROOT / "result" / "pagerank_nibble_summary.csv"
DEFAULT_GRAPH_JSON = REPO_ROOT / "result" / "pagerank_nibble_graph.json"


@dataclass(frozen=True)
class CommunityInfo:
    community_id: int | None
    community_name: str
    community_size: int


@dataclass
class NibbleRun:
    seed: str
    ppr: dict[str, float]
    residual: dict[str, float]
    push_count: int
    ranked_nodes: list[str]
    best_size: int
    conductance: float
    cut: float
    volume: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PageRank Nibble / approximate personalized PageRank on the "
            "weighted subreddit graph and export local-cluster results."
        ),
    )
    parser.add_argument("--graph-csv", type=Path, default=DEFAULT_GRAPH_CSV)
    parser.add_argument("--community-csv", type=Path, default=DEFAULT_COMMUNITY_CSV)
    parser.add_argument("--cluster-names-csv", type=Path, default=DEFAULT_CLUSTER_NAMES_CSV)
    parser.add_argument("--bridge-csv", type=Path, default=DEFAULT_BRIDGE_CSV)
    parser.add_argument("--gateway-csv", type=Path, default=DEFAULT_GATEWAY_CSV)
    parser.add_argument("--highway-csv", type=Path, default=DEFAULT_HIGHWAY_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_RESULT_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--graph-json", type=Path, default=DEFAULT_GRAPH_JSON)
    parser.add_argument(
        "--min-similarity",
        default="0.7361",
        help='Minimum edge weight to keep, or "auto" to use --percentile.',
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.97,
        help="Percentile used when --min-similarity=auto.",
    )
    parser.add_argument("--alpha", type=float, default=0.15, help="Teleport probability.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="Push threshold on residual / weighted degree.",
    )
    parser.add_argument("--seed-count", type=int, default=12)
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated seed subreddits. When omitted, seeds come from bridge/gateway/highway results.",
    )
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--max-cluster-size", type=int, default=80)
    parser.add_argument("--sweep-limit", type=int, default=500)
    parser.add_argument("--max-pushes", type=int, default=1_000_000)
    parser.add_argument("--graph-node-limit", type=int, default=220)
    parser.add_argument("--graph-edge-limit", type=int, default=1200)
    return parser.parse_args()


def detect_edge_columns(fieldnames: list[str] | None) -> tuple[str, str, str]:
    headers = set(fieldnames or [])
    if {"Subreddit_A", "Subreddit_B", "Similarity_Score"} <= headers:
        return "Subreddit_A", "Subreddit_B", "Similarity_Score"
    if {"source_subreddit", "target_subreddit", "weight"} <= headers:
        return "source_subreddit", "target_subreddit", "weight"
    raise ValueError(f"Unsupported graph CSV columns: {fieldnames}")


def safe_float(value: object, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def safe_int(value: object, fallback: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return fallback


def round_float(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def compute_percentile_threshold(csv_path: Path, percentile: float) -> float:
    scores: list[float] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _, _, weight_column = detect_edge_columns(reader.fieldnames)
        for row in reader:
            score = safe_float(row.get(weight_column), float("nan"))
            if math.isfinite(score):
                scores.append(score)

    if not scores:
        raise ValueError(f"No numeric edge weights found in {csv_path}")

    scores.sort()
    index = int((len(scores) - 1) * percentile)
    return scores[index]


def load_weighted_graph(
    csv_path: Path,
    min_weight: float,
) -> tuple[dict[str, dict[str, float]], dict[str, float], int]:
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    edge_count = 0

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        source_column, target_column, weight_column = detect_edge_columns(reader.fieldnames)
        for row in reader:
            source = str(row.get(source_column, "")).strip()
            target = str(row.get(target_column, "")).strip()
            weight = safe_float(row.get(weight_column), float("nan"))
            if not source or not target or source == target or not math.isfinite(weight):
                continue
            if weight < min_weight:
                continue

            previous = adjacency[source].get(target)
            if previous is None:
                adjacency[source][target] = weight
                adjacency[target][source] = weight
                edge_count += 1
            elif weight > previous:
                adjacency[source][target] = weight
                adjacency[target][source] = weight

    degree = {node: sum(neighbors.values()) for node, neighbors in adjacency.items()}
    return dict(adjacency), degree, edge_count


def read_cluster_names(csv_path: Path) -> dict[int, str]:
    if not csv_path.exists():
        return {}

    names: dict[int, str] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            community_id = safe_int(row.get("community_id"), -1)
            if community_id >= 0:
                names[community_id] = row.get("name") or f"Community {community_id}"
    return names


def read_community_info(
    community_csv: Path,
    cluster_names_csv: Path,
) -> dict[str, CommunityInfo]:
    names = read_cluster_names(cluster_names_csv)
    info: dict[str, CommunityInfo] = {}
    if not community_csv.exists():
        return info

    with community_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subreddit = str(row.get("subreddit", "")).strip()
            if not subreddit:
                continue
            community_id = safe_int(row.get("community_id"), -1)
            community_size = safe_int(row.get("community_size"), 0)
            info[subreddit] = CommunityInfo(
                community_id=community_id if community_id >= 0 else None,
                community_name=names.get(community_id, f"Community {community_id}"),
                community_size=community_size,
            )
    return info


def read_sorted_values(csv_path: Path, score_column: str, value_column: str = "subreddit") -> list[str]:
    if not csv_path.exists():
        return []

    rows = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = str(row.get(value_column, "")).strip()
            if value:
                rows.append((safe_float(row.get(score_column)), value))
    rows.sort(reverse=True)
    return [value for _, value in rows]


def parse_highway_nodes(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*(?:->|→|â†’)\s*", value or "") if part.strip()]


def read_highway_candidates(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return []

    candidates: list[str] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = sorted(reader, key=lambda row: safe_int(row.get("rank"), 10**9))
        for row in rows:
            candidates.extend(parse_highway_nodes(row.get("highway_nodes", "")))
    return candidates


def round_robin(lists: list[list[str]]) -> Iterable[str]:
    max_length = max((len(items) for items in lists), default=0)
    for index in range(max_length):
        for items in lists:
            if index < len(items):
                yield items[index]


def select_seeds(args: argparse.Namespace, graph_nodes: set[str], degree: dict[str, float]) -> list[str]:
    requested = [seed.strip() for seed in args.seeds.split(",") if seed.strip()]
    if requested:
        return [seed for seed in requested if seed in graph_nodes][: args.seed_count]

    bridge_candidates = read_sorted_values(args.bridge_csv, "bridge_score")
    gateway_candidates = read_sorted_values(args.gateway_csv, "gateway_score_normalized")
    highway_candidates = read_highway_candidates(args.highway_csv)
    degree_candidates = [node for node, _ in sorted(degree.items(), key=lambda item: item[1], reverse=True)]

    selected: list[str] = []
    seen: set[str] = set()
    for candidate in round_robin([bridge_candidates, gateway_candidates, highway_candidates, degree_candidates]):
        if candidate not in graph_nodes or candidate in seen:
            continue
        selected.append(candidate)
        seen.add(candidate)
        if len(selected) >= args.seed_count:
            break
    return selected


def approximate_personalized_pagerank(
    adjacency: dict[str, dict[str, float]],
    degree: dict[str, float],
    seed: str,
    alpha: float,
    epsilon: float,
    max_pushes: int,
) -> tuple[dict[str, float], dict[str, float], int]:
    if seed not in adjacency:
        raise ValueError(f"Seed not found in graph: {seed}")

    ppr: dict[str, float] = defaultdict(float)
    residual: dict[str, float] = defaultdict(float)
    residual[seed] = 1.0
    heap: list[tuple[float, str]] = []

    def priority(node: str) -> float:
        node_degree = degree.get(node, 0.0)
        return residual[node] / node_degree if node_degree > 0 else 0.0

    heapq.heappush(heap, (-priority(seed), seed))
    push_count = 0

    while heap and push_count < max_pushes:
        _, node = heapq.heappop(heap)
        if priority(node) <= epsilon:
            continue

        node_residual = residual[node]
        ppr[node] += alpha * node_residual
        remaining = (1.0 - alpha) * node_residual
        residual[node] = remaining / 2.0
        push_count += 1

        node_degree = degree[node]
        neighbor_mass = remaining / 2.0
        for neighbor, weight in adjacency[node].items():
            before = priority(neighbor)
            residual[neighbor] += neighbor_mass * weight / node_degree
            if before <= epsilon < priority(neighbor):
                heapq.heappush(heap, (-priority(neighbor), neighbor))

        if priority(node) > epsilon:
            heapq.heappush(heap, (-priority(node), node))

    return dict(ppr), dict(residual), push_count


def sweep_best_cluster(
    adjacency: dict[str, dict[str, float]],
    degree: dict[str, float],
    ppr: dict[str, float],
    min_cluster_size: int,
    max_cluster_size: int,
    sweep_limit: int,
) -> tuple[list[str], float, float, float]:
    order = [
        node
        for node, score in sorted(
            ppr.items(),
            key=lambda item: item[1] / max(degree.get(item[0], 0.0), float("1e-300")),
            reverse=True,
        )
        if score > 0 and degree.get(node, 0.0) > 0
    ][:sweep_limit]

    total_volume = sum(degree.values())
    in_prefix: set[str] = set()
    cut = 0.0
    volume = 0.0
    best: tuple[int, float, float, float] | None = None

    for index, node in enumerate(order, start=1):
        in_prefix.add(node)
        volume += degree[node]
        for neighbor, weight in adjacency[node].items():
            if neighbor in in_prefix:
                cut -= weight
            else:
                cut += weight

        if index < min_cluster_size or index > max_cluster_size:
            continue

        denominator = min(volume, total_volume - volume)
        conductance = cut / denominator if denominator > 0 else 1.0
        if best is None or conductance < best[1]:
            best = (index, conductance, cut, volume)

    if best is None:
        fallback_size = min(max(len(order), 1), max_cluster_size)
        return order[:fallback_size], 1.0, 0.0, sum(degree.get(node, 0.0) for node in order[:fallback_size])

    best_size, best_conductance, best_cut, best_volume = best
    return order[:best_size], best_conductance, best_cut, best_volume


def run_nibble_for_seeds(
    adjacency: dict[str, dict[str, float]],
    degree: dict[str, float],
    seeds: list[str],
    args: argparse.Namespace,
) -> list[NibbleRun]:
    runs: list[NibbleRun] = []
    for seed in seeds:
        ppr, residual, push_count = approximate_personalized_pagerank(
            adjacency=adjacency,
            degree=degree,
            seed=seed,
            alpha=args.alpha,
            epsilon=args.epsilon,
            max_pushes=args.max_pushes,
        )
        ranked_nodes, conductance, cut, volume = sweep_best_cluster(
            adjacency=adjacency,
            degree=degree,
            ppr=ppr,
            min_cluster_size=args.min_cluster_size,
            max_cluster_size=args.max_cluster_size,
            sweep_limit=args.sweep_limit,
        )
        runs.append(
            NibbleRun(
                seed=seed,
                ppr=ppr,
                residual=residual,
                push_count=push_count,
                ranked_nodes=ranked_nodes,
                best_size=len(ranked_nodes),
                conductance=conductance,
                cut=cut,
                volume=volume,
            ),
        )
    return runs


def get_info(community_info: dict[str, CommunityInfo], subreddit: str) -> CommunityInfo:
    return community_info.get(subreddit, CommunityInfo(None, "", 0))


def write_outputs(
    runs: list[NibbleRun],
    community_info: dict[str, CommunityInfo],
    degree: dict[str, float],
    args: argparse.Namespace,
    min_weight: float,
) -> None:
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    result_fields = [
        "seed_subreddit",
        "rank",
        "subreddit",
        "community_id",
        "community_name",
        "ppr_score",
        "score_per_degree",
        "weighted_degree",
        "cluster_size",
        "cluster_conductance",
        "cluster_cut",
        "cluster_volume",
        "alpha",
        "epsilon",
        "min_edge_weight",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields)
        writer.writeheader()
        for run in runs:
            for rank, subreddit in enumerate(run.ranked_nodes, start=1):
                info = get_info(community_info, subreddit)
                weighted_degree = degree.get(subreddit, 0.0)
                ppr_score = run.ppr.get(subreddit, 0.0)
                writer.writerow(
                    {
                        "seed_subreddit": run.seed,
                        "rank": rank,
                        "subreddit": subreddit,
                        "community_id": "" if info.community_id is None else info.community_id,
                        "community_name": info.community_name,
                        "ppr_score": f"{ppr_score:.12g}",
                        "score_per_degree": f"{(ppr_score / weighted_degree if weighted_degree else 0.0):.12g}",
                        "weighted_degree": f"{weighted_degree:.12g}",
                        "cluster_size": run.best_size,
                        "cluster_conductance": f"{run.conductance:.12g}",
                        "cluster_cut": f"{run.cut:.12g}",
                        "cluster_volume": f"{run.volume:.12g}",
                        "alpha": args.alpha,
                        "epsilon": args.epsilon,
                        "min_edge_weight": f"{min_weight:.12g}",
                    },
                )

    summary_fields = [
        "seed_subreddit",
        "seed_community_id",
        "seed_community_name",
        "cluster_size",
        "cluster_conductance",
        "cluster_cut",
        "cluster_volume",
        "ppr_mass",
        "residual_mass",
        "push_count",
        "ranked_node_count",
        "top_subreddits",
        "top_communities",
        "alpha",
        "epsilon",
        "min_edge_weight",
    ]
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for run in runs:
            seed_info = get_info(community_info, run.seed)
            top_nodes = run.ranked_nodes[:12]
            top_communities = []
            seen_communities = set()
            for node in run.ranked_nodes:
                info = get_info(community_info, node)
                if info.community_id is None or info.community_id in seen_communities:
                    continue
                seen_communities.add(info.community_id)
                top_communities.append(info.community_name or f"Community {info.community_id}")
                if len(top_communities) >= 6:
                    break

            writer.writerow(
                {
                    "seed_subreddit": run.seed,
                    "seed_community_id": "" if seed_info.community_id is None else seed_info.community_id,
                    "seed_community_name": seed_info.community_name,
                    "cluster_size": run.best_size,
                    "cluster_conductance": f"{run.conductance:.12g}",
                    "cluster_cut": f"{run.cut:.12g}",
                    "cluster_volume": f"{run.volume:.12g}",
                    "ppr_mass": f"{sum(run.ppr.values()):.12g}",
                    "residual_mass": f"{sum(run.residual.values()):.12g}",
                    "push_count": run.push_count,
                    "ranked_node_count": len(run.ppr),
                    "top_subreddits": " | ".join(top_nodes),
                    "top_communities": " | ".join(top_communities),
                    "alpha": args.alpha,
                    "epsilon": args.epsilon,
                    "min_edge_weight": f"{min_weight:.12g}",
                },
            )


def build_graph_payload(
    runs: list[NibbleRun],
    adjacency: dict[str, dict[str, float]],
    degree: dict[str, float],
    community_info: dict[str, CommunityInfo],
    args: argparse.Namespace,
    min_weight: float,
) -> dict[str, object]:
    selected: list[str] = []
    owners: dict[str, list[str]] = defaultdict(list)
    seed_set = {run.seed for run in runs}

    for run in runs:
        for node in run.ranked_nodes:
            if len(selected) >= args.graph_node_limit and node not in selected:
                continue
            if node not in selected:
                selected.append(node)
            owners[node].append(run.seed)

    selected_set = set(selected)
    max_ppr_by_node: dict[str, float] = defaultdict(float)
    for run in runs:
        for node in selected_set:
            max_ppr_by_node[node] = max(max_ppr_by_node[node], run.ppr.get(node, 0.0))

    max_ppr = max(max_ppr_by_node.values(), default=1.0)
    nodes = []
    for node in selected:
        info = get_info(community_info, node)
        score = max_ppr_by_node[node]
        owner_text = ", ".join(owners[node][:4])
        community_label = info.community_name or "Unknown community"
        nodes.append(
            {
                "id": node,
                "label": node,
                "group": "seed" if node in seed_set else f"cluster:{owners[node][0]}",
                "shape": "star" if node in seed_set else "dot",
                "size": round(9 + 28 * math.sqrt(score / max_ppr), 2) if max_ppr > 0 else 9,
                "value": round_float(score, 8),
                "title": (
                    f"Subreddit: {node}<br>"
                    f"Role: {'Seed' if node in seed_set else 'PageRank Nibble member'}<br>"
                    f"Seed(s): {owner_text}<br>"
                    f"Community: {community_label}<br>"
                    f"PPR score: {score:.8f}<br>"
                    f"Weighted degree: {degree.get(node, 0.0):.4f}"
                ),
            },
        )

    edges = []
    seen_edges: set[tuple[str, str]] = set()
    for source in selected:
        for target, weight in adjacency.get(source, {}).items():
            if target not in selected_set:
                continue
            edge_key = tuple(sorted((source, target)))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append((source, target, weight))

    edges.sort(key=lambda item: item[2], reverse=True)
    edge_payload = [
        {
            "id": f"ppr:{source}:{target}",
            "from": source,
            "to": target,
            "width": round(0.7 + 4.3 * ((weight - min_weight) / max(1.0 - min_weight, 1e-9)), 3),
            "Similarity_Score": round_float(weight, 6),
            "title": f"Similarity: {weight:.4f}",
        }
        for source, target, weight in edges[: args.graph_edge_limit]
    ]

    return {
        "id": "pagerank_nibble",
        "title": "PageRank Nibble local clusters",
        "description": "Approximate Personalized PageRank neighborhoods from bridge/gateway/highway seeds.",
        "sourceFile": args.output_csv.relative_to(REPO_ROOT).as_posix(),
        "extractedFromHtml": False,
        "parameters": {
            "alpha": args.alpha,
            "epsilon": args.epsilon,
            "minEdgeWeight": min_weight,
            "seedCount": len(runs),
        },
        "nodes": nodes,
        "edges": edge_payload,
        "groups": sorted({node["group"] for node in nodes}),
    }


def write_graph_json(
    runs: list[NibbleRun],
    adjacency: dict[str, dict[str, float]],
    degree: dict[str, float],
    community_info: dict[str, CommunityInfo],
    args: argparse.Namespace,
    min_weight: float,
) -> None:
    args.graph_json.parent.mkdir(parents=True, exist_ok=True)
    payload = build_graph_payload(runs, adjacency, degree, community_info, args, min_weight)
    args.graph_json.write_text(f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.min_similarity.lower() == "auto":
        min_weight = compute_percentile_threshold(args.graph_csv, args.percentile)
    else:
        min_weight = safe_float(args.min_similarity, 0.0)

    print(f"Loading graph from {args.graph_csv} with min edge weight {min_weight:.4f}")
    adjacency, degree, edge_count = load_weighted_graph(args.graph_csv, min_weight)
    if not adjacency:
        raise RuntimeError("Graph is empty after filtering.")

    community_info = read_community_info(args.community_csv, args.cluster_names_csv)
    seeds = select_seeds(args, set(adjacency), degree)
    if not seeds:
        raise RuntimeError("No valid PageRank Nibble seeds found.")

    print(f"Graph: {len(adjacency):,} nodes, {edge_count:,} undirected edges")
    print(f"Seeds: {', '.join(seeds)}")
    runs = run_nibble_for_seeds(adjacency, degree, seeds, args)

    write_outputs(runs, community_info, degree, args, min_weight)
    write_graph_json(runs, adjacency, degree, community_info, args, min_weight)

    best = min(runs, key=lambda run: run.conductance)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.graph_json}")
    print(
        "Best local cluster: "
        f"seed={best.seed}, size={best.best_size}, conductance={best.conductance:.4f}"
    )


if __name__ == "__main__":
    main()

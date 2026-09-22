"""Stream candidate family assignments; preserve code holds across text links."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from speck.data.code_families import partition_code_families
from speck.data.joint_graph import _bound_json, _Families
from speck.provenance.io import file_sha256


def _digest(members):
    return hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()


def _partition(seed, identity):
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10000
    return "final" if bucket < 500 else "development" if bucket < 1000 else "train"


def write_partitions(plan, sources, edges, output, *, firewall_matches=()):
    """Assign whole connected components, then stream singleton documents in source order.

    Source-qualified content nodes prevent accidental joins; explicit exact/near edges and
    repository identities supply the links. A component's sorted unique content hashes and
    repository names define its identity. Neither source order nor duplicate copies affect it.
    """
    rule = plan["partition"]
    if (
        not isinstance(rule.get("seed"), str)
        or not rule["seed"]
        or [rule.get(key) for key in ("train_buckets", "development_buckets", "final_buckets")]
        != [9000, 500, 500]
        or rule.get("total_buckets") != 10000
        or rule.get("identity") != "sha256_sorted_unique_content_and_repository_nodes_v1"
        or rule.get("firewall") != "exclude_primary_and_unseen_from_all_candidate_partitions"
    ):
        raise ValueError("unsupported frozen family partition rule")
    families = _Families()
    for first, second, _, _ in edges:
        families.union(first, second)
    holds = defaultdict(set)
    for match in firewall_matches:
        holds[("code_cohort", match["content_sha256"])].add("firewall_reference_overlap")
    code_rows = {}
    if "code_cohort" in plan:
        _, graph = _bound_json(plan["code_cohort"]["family_inputs"], "code family inputs")
        with Path(plan["code_cohort"]["documents"]["path"]).open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["record_id"] in code_rows:
                    raise ValueError("duplicate code record id")
                code_rows[row["record_id"]] = row
        graph_rows = {row["id"]: row for row in graph["records"]}
        if code_rows.keys() - graph_rows.keys() or any(
            row.get("duplicate_group") and key not in code_rows for key, row in graph_rows.items()
        ):
            raise ValueError("code documents and family inventory differ")
        assignments = partition_code_families(
            graph["records"],
            aliases=graph["aliases"],
            held_repositories=graph["held_repositories"],
            seed=rule["seed"],
        )
        for row in assignments:
            if row["id"] not in code_rows:
                continue  # Evidence-backed external parent nodes carry no sampled text.
            content = code_rows[row["id"]]["released_content_sha256"]
            if graph_rows[row["id"]].get("duplicate_group") != content:
                raise ValueError("code family content identity differs")
            node = ("code_cohort", content)
            families.find(node)
            for repository in row["repositories"]:
                families.union(node, ("@repository", repository))
            holds[node].update(row["hold_reasons"])
    members = defaultdict(set)
    reasons = defaultdict(set)
    for node in list(families.parent):
        root = families.find(node)
        members[root].add(("repo:" if node[0] == "@repository" else "sha256:") + node[1])
        reasons[root].update(holds[node])
    identities = {root: _digest(values) for root, values in members.items()}
    wanted = {node for node in families.parent if node[0] != "@repository"}
    seen = set()
    counts = defaultdict(Counter)
    tokens = defaultdict(Counter)
    path = output / "partitions.jsonl"
    with path.open("w") as handle:
        for source in sources:
            if source["id"].startswith("@"):
                raise ValueError("reserved source id")
            with source["documents"].open() as documents:
                for ordinal, line in enumerate(documents):
                    row = json.loads(line)
                    content = row["released_content_sha256"]
                    node = (source["id"], content)
                    hold = []
                    if node in families.parent:
                        seen.add(node)
                        root = families.find(node)
                        identity = identities[root]
                        hold = sorted(reasons[root])
                    else:
                        identity = _digest(["sha256:" + content])
                    partition = "quarantine" if hold else _partition(rule["seed"], identity)
                    count = row["token_count"]
                    if type(count) is not int or count < 1:
                        raise ValueError("invalid document token count")
                    counts[source["id"]][partition] += 1
                    tokens[source["id"]][partition] += count
                    assignment = {
                        "source": source["id"],
                        "ordinal": ordinal,
                        "released_content_sha256": content,
                        "token_count": count,
                        "family_component_sha256": identity,
                        "candidate_partition": partition,
                        "hold_reasons": hold,
                        "training_admitted": False,
                    }
                    if source["id"] == "code_cohort":
                        assignment["record_id"] = row["record_id"]
                    handle.write(
                        json.dumps(assignment, sort_keys=True, separators=(",", ":")) + "\n"
                    )
    if wanted - seen:
        raise ValueError("graph contains documents missing from the partition inventory")
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "rule": rule,
        "documents": {key: dict(value) for key, value in sorted(counts.items())},
        "tokens": {key: dict(value) for key, value in sorted(tokens.items())},
        "boundary": "Candidate inventory only. Code coverage is the pinned review cohort, not the full retained code stock. All other eligibility gates remain required.",
    }

"""Assign code families before selection; a split assignment never admits training data."""

import hashlib
import re


def repository_key(value):
    """Accept an explicit owner/repository identity, never guess a fork's origin."""
    if not isinstance(value, str):
        raise ValueError("repository identity must be a string")
    value = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]+/[a-z0-9_.-]+", value):
        raise ValueError("expected owner/repository identity")
    return value


class Families:
    """Union-find over hashable nodes whose root is the smallest member."""

    def __init__(self):
        self.parent = {}

    def find(self, node):
        self.parent.setdefault(node, node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[max(first, second)] = min(first, second)


def partition_code_families(records, *, aliases=(), held_repositories=(), seed):
    """Join repositories, declared aliases, parents and duplicate groups transitively.

    Input fields: id, repository (or None), parents, duplicate_group and benchmark_overlap.
    Aliases must come from recorded origin evidence, not name similarity. Missing origins or
    parents quarantine the whole component. A benchmark match holds the whole component.
    Call on the complete frozen candidate inventory; adding links can change assignments.
    """
    if not isinstance(seed, str) or not seed:
        raise ValueError("a nonempty frozen seed is required")
    records = list(records)
    by_id = {}
    links = Families()
    find, union = links.find, links.union

    for row in records:
        key = row.get("id")
        if not isinstance(key, str) or not key or key in by_id:
            raise ValueError("record IDs must be unique nonempty strings")
        by_id[key] = row
        find(("record", key))
    for a, b in aliases:
        union(("repo", repository_key(a)), ("repo", repository_key(b)))
    unresolved = set()
    overlaps = set()
    for key, row in by_id.items():
        node = ("record", key)
        repo = row.get("repository")
        if repo:
            union(node, ("repo", repository_key(repo)))
        else:
            unresolved.add(node)
        parents = row.get("parents", [])
        if not isinstance(parents, list) or any(not isinstance(p, str) for p in parents):
            raise ValueError("parents must be a list of record IDs")
        for parent in parents:
            if parent not in by_id:
                unresolved.add(node)
            else:
                union(node, ("record", parent))
        duplicate = row.get("duplicate_group")
        if duplicate is not None:
            if not isinstance(duplicate, str) or not duplicate:
                raise ValueError("duplicate group must be a nonempty string")
            union(node, ("duplicate", duplicate))
        overlap = row.get("benchmark_overlap", False)
        if type(overlap) is not bool:
            raise ValueError("benchmark_overlap must be boolean")
        if overlap:
            overlaps.add(node)
    held = {("repo", repository_key(r)) for r in held_repositories}
    held_roots = {find(node) for node in held | overlaps}
    unresolved_roots = {find(node) for node in unresolved}
    families = {}
    for node in list(links.parent):
        if node[0] == "repo":
            families.setdefault(find(node), set()).add(node[1])
    result = []
    for key in sorted(by_id):
        component = find(("record", key))
        members = sorted(families.get(component, set()))
        identity = hashlib.sha256("\n".join(members).encode()).hexdigest()
        reasons = []
        if component in held_roots:
            reasons.append("benchmark_family_or_content_overlap")
        if component in unresolved_roots:
            reasons.append("unresolved_origin_or_parent")
        if reasons:
            partition = "quarantine"
        else:
            digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
            bucket = int.from_bytes(digest[:8], "big") % 10000
            partition = "final" if bucket < 500 else "development" if bucket < 1000 else "train"
        result.append(
            {
                "id": key,
                "family_component_sha256": identity,
                "repositories": members,
                "candidate_partition": partition,
                "hold_reasons": reasons,
                "training_admitted": False,
            }
        )
    return result

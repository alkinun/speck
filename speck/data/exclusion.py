"""Verify retained reference exclusion without importing acquisition workflows."""


def verify_excluded_parent(parent, references):
    """Check exact reference identities/precedence and preservation in a dedup result."""

    if (
        parent.get("sources", [])[:12] != references["sources"]
        or parent.get("policy") != references["policy"]
        or parent.get("gates", {}).get("global_exact_deduplication") != "pass"
        or parent.get("gates", {}).get(
            "disk_backed_Minhash_candidates_and_verified_near_deduplication"
        )
        != "pass"
    ):
        raise ValueError("excluded parent does not bind the complete reference pass")
    for source in references["sources"]:
        if parent.get("outputs", {}).get(source["id"], {}).get("sha256") != source["sha256"]:
            raise ValueError("a firewall reference was removed or changed during the pass")

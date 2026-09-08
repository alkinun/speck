import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
FLAGSHIP = ROOT / "research/flagship"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_v3_plan_corrects_only_embedding_head_accounting_and_binds_the_decision():
    plan = json.loads((FLAGSHIP / "tokenizer_plan_v3.json").read_text())
    predecessor = json.loads((FLAGSHIP / "tokenizer_plan_v2.json").read_text())
    contract = json.loads((FLAGSHIP / "embedding_head_contract_v1.json").read_text())

    assert plan["format_version"] == 3
    assert _sha256(ROOT / plan["supersedes"]["path"]) == plan["supersedes"]["sha256"]
    assert (
        _sha256(ROOT / plan["embedding_head_contract"]["path"])
        == plan["embedding_head_contract"]["sha256"]
    )
    assert plan["candidates"] == predecessor["candidates"]
    assert plan["trainer"] == predecessor["trainer"]
    assert plan["sample"] == predecessor["sample"]
    assert plan["model_accounting"] == {
        "embedding_width": 2048,
        "tied_embeddings": True,
        "architecture_config_tie_word_embeddings": True,
        "physical_embedding_and_head_parameter_formula": ("effective_vocab_size * embedding_width"),
        "chat_added_tokens": 3,
        "packed_token_dtype": "uint16",
    }
    assert contract["decision"]["tie_word_embeddings"] is True
    assert contract["decision"]["physical_parameter_objects"] == 1
    assert contract["decision"]["untied_support"] == "unsupported"
    assert contract["gpu_ablation_performed"] is False

    width = plan["model_accounting"]["embedding_width"]
    chat = plan["model_accounting"]["chat_added_tokens"]
    assert [(vocab + chat) * width for vocab in (32_000, 32_768, 40_960)] == [
        65_542_144,
        67_115_008,
        83_892_224,
    ]

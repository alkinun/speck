import json

from scripts.helmet_multilexsum_decision import (
    canonical_visible_text,
    protocol_identity,
    unseeded_train_shuffle_count,
)


def test_unseeded_train_shuffle_is_distinct_from_seeded_shuffle():
    source = "train_data.shuffle(); train_data.shuffle(seed=42); other.shuffle()"
    assert unseeded_train_shuffle_count(source) == 1


def test_visible_text_ignores_script_transport_noise():
    assert canonical_visible_text("<p>Terms</p><script>one</script>") == canonical_visible_text(
        "<p>Terms</p><script>two</script>"
    )


def test_multilexsum_protocol_identity_excludes_registration_fields():
    frozen = {"format": "protocol", "status": "frozen", "source": {"id": "multi"}}
    executed = json.loads(json.dumps(frozen))
    executed.update({"status": "executed", "result": {"sha256": "abc"}})
    assert protocol_identity(frozen) == protocol_identity(executed)

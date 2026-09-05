import json

from scripts.helmet_narrativeqa_decision import (
    canonical_visible_text,
    protocol_identity,
    unseeded_shuffle_count,
)


def test_unseeded_shuffle_is_scoped_to_loader_function():
    source = '''
def load_narrativeqa():
    data.shuffle(seed=42)
    demos.shuffle()

def other():
    data.shuffle()
'''
    assert unseeded_shuffle_count(source, "load_narrativeqa") == 1


def test_visible_text_ignores_script_transport_noise():
    assert canonical_visible_text("<p>Rights</p><script>one</script>") == canonical_visible_text(
        "<p>Rights</p><script>two</script>"
    )


def test_narrativeqa_protocol_identity_excludes_registration_fields():
    frozen = {"format": "protocol", "status": "frozen", "source": {"id": "narrative"}}
    executed = json.loads(json.dumps(frozen))
    executed.update({"status": "executed", "result": {"sha256": "abc"}})
    assert protocol_identity(frozen) == protocol_identity(executed)

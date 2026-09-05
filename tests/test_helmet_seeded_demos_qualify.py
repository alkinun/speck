import json

from scripts.helmet_seeded_demos_qualify import protocol_identity, target_unseeded_shuffles


def test_target_unseeded_shuffles_tracks_only_two_loaders():
    source = '''
def load_narrativeqa():
    train.shuffle()
    train.shuffle(seed=42)

def load_multi_lexsum():
    train.shuffle()

def other():
    train.shuffle()
'''
    assert target_unseeded_shuffles(source) == {
        "load_narrativeqa": 1,
        "load_multi_lexsum": 1,
    }


def test_seeded_demo_protocol_identity_excludes_registration_fields():
    frozen = {"format": "protocol", "status": "frozen", "patch": {"sha256": "abc"}}
    executed = json.loads(json.dumps(frozen))
    executed.update({"status": "executed", "result": {"sha256": "def"}})
    assert protocol_identity(frozen) == protocol_identity(executed)

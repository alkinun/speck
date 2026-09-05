import json

from scripts.helmet_infinitebench_decision import protocol_identity, unseeded_shuffle_count


def test_seeded_shuffle_is_not_counted_as_unseeded():
    source = '''
def load_infbench():
    data.shuffle(seed=42)
    data.filter().shuffle(seed=7)

def other():
    data.shuffle()
'''
    assert unseeded_shuffle_count(source, "load_infbench") == 0


def test_infinitebench_protocol_identity_excludes_registration_fields():
    frozen = {"format": "protocol", "status": "frozen", "source": {"id": "infinite"}}
    executed = json.loads(json.dumps(frozen))
    executed.update({"status": "executed", "result": {"sha256": "abc"}})
    assert protocol_identity(frozen) == protocol_identity(executed)

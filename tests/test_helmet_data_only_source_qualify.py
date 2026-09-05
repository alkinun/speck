import json

from scripts.helmet_data_only_source_qualify import protocol_identity, verify_upstream_rows


class IntentFeature:
    names = ["known", "oos"]


class Features(dict):
    def __getitem__(self, key):
        assert key == "intent"
        return IntentFeature()


class Dataset:
    features = Features()

    def __iter__(self):
        return iter([{"text": "one", "intent": 0}, {"text": "two", "intent": 1}])


def test_verify_upstream_rows_requires_declared_concatenation_order():
    upstream = {"in": [["one", "known"]], "out": [["two", "oos"]]}
    assert verify_upstream_rows(Dataset(), upstream, ["in", "out"])[
        "all_text_and_intent_values_match_in_order"
    ]


def test_source_protocol_identity_excludes_registration_fields():
    frozen = {"format": "protocol", "status": "frozen", "source": {"id": "clinc"}}
    executed = json.loads(json.dumps(frozen))
    executed.update({"status": "executed", "result": {"sha256": "abc"}})
    assert protocol_identity(frozen) == protocol_identity(executed)

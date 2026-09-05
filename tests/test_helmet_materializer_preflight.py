import json

from scripts.helmet_materializer_preflight import canonical_rows, protocol_identity


class Rows:
    def __iter__(self):
        return iter([{"label": 1, "text": "one"}, {"text": "two", "label": 0}])


def test_canonical_rows_is_key_order_independent_and_row_order_sensitive():
    first = canonical_rows(Rows())

    class ReorderedKeys:
        def __iter__(self):
            return iter([{"text": "one", "label": 1}, {"label": 0, "text": "two"}])

    class ReorderedRows:
        def __iter__(self):
            return iter([{"label": 0, "text": "two"}, {"label": 1, "text": "one"}])

    assert canonical_rows(ReorderedKeys()) == first
    assert canonical_rows(ReorderedRows()) != first


def test_materializer_protocol_identity_excludes_registration_fields():
    frozen = {"format": "protocol", "status": "frozen", "scope": {"datasets": 2}}
    executed = json.loads(json.dumps(frozen))
    executed.update({"status": "executed", "result": {"sha256": "abc"}})
    assert protocol_identity(frozen) == protocol_identity(executed)

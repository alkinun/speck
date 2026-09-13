"""Keep software, frozen-record, and local integration checks explicit."""


def pytest_addoption(parser):
    parser.addoption(
        "--evidence", action="store_true", help="include frozen research record checks"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        help="include checks requiring local artifacts/hardware",
    )


def pytest_collection_modifyitems(config, items):
    selected, deselected = [], []
    for item in items:
        include = all(
            not item.get_closest_marker(marker) or config.getoption(f"--{marker}")
            for marker in ("evidence", "integration")
        )
        (selected if include else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected

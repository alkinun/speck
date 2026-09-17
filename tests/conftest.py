"""Keep optional frozen-record checks explicit."""


def pytest_addoption(parser):
    parser.addoption(
        "--evidence", action="store_true", help="include frozen research record checks"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--evidence"):
        return
    selected, deselected = [], []
    for item in items:
        (deselected if item.get_closest_marker("evidence") else selected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected

"""Run one disposable systemd ReadOnlyPaths qualification fixture."""

import json

from speck.paper_finalist_systems_sandbox import qualify_disposable_sandbox


def main():
    print(json.dumps(qualify_disposable_sandbox(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Select, jointly exclude, and pack the finite retained-data pilot."""

import argparse

from speck.data.pilot import exclude_candidates, pack_candidates, select_candidates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--stage", choices=("select", "exclude", "pack"), required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--runtime-root")
    parser.add_argument("--evaluation")
    args = parser.parse_args(argv)
    if args.stage == "select":
        if not args.runtime_root or not args.evaluation:
            parser.error("selection requires --runtime-root and --evaluation")
        select_candidates(args.experiment, args.runtime_root, args.work_dir, args.evaluation)
    elif args.stage == "exclude":
        exclude_candidates(args.work_dir)
    else:
        pack_candidates(args.experiment, args.work_dir)


if __name__ == "__main__":
    main()

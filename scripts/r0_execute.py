"""Check a bound R0 request, or explicitly execute one finite synthetic qualification attempt."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", nargs="?")
    parser.add_argument("--case", default="baseline-4096")
    parser.add_argument("--workers", type=int, choices=(1, 4), default=1)
    parser.add_argument("--allocated-gpus", type=int)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--ledger")
    parser.add_argument("--prior-r0-gpu-hours", type=float)
    parser.add_argument("--worker-request", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-phase",
        choices=("single", "initial", "restart"),
        default="single",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.worker_request:
        from speck.operations.r0_worker import main as worker_main

        worker_main(args.worker_request, args.worker_phase)
        return
    if not args.plan or not args.case or args.allocated_gpus is None:
        parser.error("plan, --case and --allocated-gpus are required")
    from speck.operations.r0_executor import prepare_request, run_attempt

    request = prepare_request(args.plan, args.case, args.workers, args.allocated_gpus)
    if args.run:
        if not args.ledger or args.prior_r0_gpu_hours is None:
            parser.error("--run requires the shared R0 --ledger and explicit --prior-r0-gpu-hours")
        result = run_attempt(request, args.ledger, args.prior_r0_gpu_hours)
        print(json.dumps(result, indent=2))
        if result["status"] not in {
            "bounded_synthetic_checks_pass",
            "bounded_fresh_process_checks_pass",
        }:
            raise SystemExit(1)
    else:
        print(
            json.dumps(
                {
                    "status": "request_bound_no_execution",
                    "request_sha256": request["request_sha256"],
                    "case": request["case"]["id"],
                    "workers": args.workers,
                    "allocated_gpus": args.allocated_gpus,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

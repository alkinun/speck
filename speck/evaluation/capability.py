"""Execute the frozen five-task capability protocol on development or final inputs."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from speck.evaluation.code_runner import check_sandbox, run_python
from speck.provenance.io import atomic_json, file_sha256

TASK_FILES = {
    "gsm8k": "gsm8k/gsm8k-cot-zeroshot.yaml",
    "ifeval": "ifeval/ifeval.yaml",
    "arc_challenge": "arc/arc_challenge.yaml",
    "hellaswag": "hellaswag/hellaswag.yaml",
}


def verify_scorers(protocol):
    versions = {}
    for package, key in (("lm_eval", "lm_eval_commit"), ("evalplus", "evalplus_commit")):
        dist = importlib.metadata.distribution(package)
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != protocol["scorers"][key]:
            raise ValueError(f"install the pinned {package} source revision")
        versions[package] = {"version": dist.version, "commit": protocol["scorers"][key]}
    return versions


def selected_rows(prepared, benchmark, partition, limit):
    import pyarrow.parquet as pq

    if file_sha256(benchmark["path"]) != benchmark["sha256"]:
        raise ValueError("benchmark bytes changed")
    wanted = set(prepared["partitions"][benchmark["id"]][partition])
    rows = []
    if benchmark["format"] == "parquet":
        source = pq.read_table(benchmark["path"]).to_pylist()
    else:
        with open(benchmark["path"]) as handle:
            source = [json.loads(line) for line in handle if line.strip()]
    if len(source) != benchmark["expected_tasks"]:
        raise ValueError("benchmark row count changed")
    for ordinal, row in enumerate(source):
        identity = str(
            ordinal if benchmark["task_id_field"] is None else row[benchmark["task_id_field"]]
        )
        if identity in wanted:
            rows.append({**row, "speck_task_id": identity})
    if len(rows) != len(wanted):
        raise ValueError("prepared task identities no longer match the source")
    return rows[:limit] if limit else rows


def frozen_task(name, rows):
    import lm_eval
    from datasets import Dataset
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks._yaml_loader import load_yaml

    class FrozenTask(ConfigurableTask):
        def download(self, *args, **kwargs):
            self.dataset = {"test": Dataset.from_list(rows)}

    config = load_yaml(
        str(Path(lm_eval.__file__).parent / "tasks" / TASK_FILES[name]),
        resolve_func=True,
        recursive=True,
    )
    config.update(
        test_split="test",
        training_split=None,
        validation_split=None,
        fewshot_split=None,
        num_fewshot=0,
    )
    return FrozenTask(config=config)


def score_text(task, doc, response):
    from lm_eval.api.instance import Instance

    instance = Instance(
        request_type="generate_until", doc=doc, arguments=("", {}), idx=0, resps=[response]
    )
    for ensemble in task._filters:
        ensemble.apply([instance])
    return {
        f"{metric},{name}": value
        for name, filtered in instance.filtered_resps.items()
        for metric, value in task.process_results(doc, [filtered]).items()
    }


def qualify(prepared, protocol):
    """Check graders with known answers, faults, and isolation probes; no model scores."""
    from evalplus.sanitize import sanitize

    from speck.evaluation.tools import golden_checks

    result = {"scorers": verify_scorers(protocol), "checks": {}}
    row = {"question": "What is two plus two?", "answer": "2 + 2 = 4. #### 4"}
    task = frozen_task("gsm8k", [row])
    assert all(score_text(task, row, "The answer is 4.").values())
    assert not any(score_text(task, row, "The answer is 5.").values())
    result["checks"]["gsm8k_correct_and_wrong"] = True
    row = {
        "key": 0,
        "prompt": "Include the word sunflower.",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["sunflower"]}],
    }
    task = frozen_task("ifeval", [row])
    correct, wrong = (
        score_text(task, row, "A sunflower grows."),
        score_text(task, row, "A rose grows."),
    )
    assert correct["prompt_level_strict_acc,none"] and not wrong["prompt_level_strict_acc,none"]
    result["checks"]["ifeval_correct_and_wrong"] = True
    for name in ("arc_challenge", "hellaswag"):
        benchmark = next(item for item in prepared["benchmarks"] if item["id"] == name)
        task = frozen_task(name, selected_rows(prepared, benchmark, "development", 1))
        doc = list(task.eval_docs)[0]
        target = task.doc_to_target(doc)
        scores = [
            (-1.0 if i == target else -1000.0, False) for i in range(len(task.doc_to_choice(doc)))
        ]
        assert all(task.process_results(doc, scores).values())
        scores = [(-1000.0 if i == target else -1.0, False) for i in range(len(scores))]
        assert not any(task.process_results(doc, scores).values())
        result["checks"][f"{name}_correct_and_wrong"] = True
    benchmark = next(item for item in prepared["benchmarks"] if item["id"] == "humanevalplus")
    checks = []
    for row in selected_rows(prepared, benchmark, "development", 0):
        code = sanitize(row["prompt"] + row["canonical_solution"], entrypoint=row["entry_point"])
        outcome = run_python(code, row["test"], row["entry_point"])
        checks.append({"task_id": row["speck_task_id"], **outcome})
    result["code_canonical"] = checks
    assert all(row["status"] == "pass" for row in checks), checks
    assert (
        run_python("def f(): return 0", "def check(f): assert f() == 1", "f")["status"] == "failed"
    )
    assert (
        run_python("def f():\n while True: pass", "def check(f): f()", "f", seconds=1)["status"]
        == "timeout"
    )
    assert (
        run_python("import os\nos._exit(0)", "def check(f): pass", "f")["status"]
        == "runner_failure"
    )
    isolation = """def f():
 import os, socket
 assert not os.path.exists('/home/alkin/.ssh')
 assert not os.path.exists('/mnt/speck-data')
 try:
  socket.create_connection(('1.1.1.1', 443), timeout=0.2)
 except OSError:
  return True
 raise AssertionError('network reachable')
"""
    assert run_python(isolation, "def check(f): assert f()", "f")["status"] == "pass"
    result["checks"]["code_fault_timeout_exit_and_isolation"] = True
    result["tool_episodes"] = golden_checks()
    result["status"] = "pass"
    return result


def evaluate(args, prepared, protocol):
    import torch
    from evalplus.sanitize import sanitize
    from lm_eval.api.instance import Instance
    from lm_eval.models.huggingface import HFLM

    scorers = verify_scorers(protocol)
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        raise ValueError("reference model revision must be a full commit hash")
    model = HFLM(
        pretrained=args.model,
        revision=args.revision,
        dtype="bfloat16",
        device=args.device,
        batch_size=1,
        max_length=4096,
        trust_remote_code=False,
    )
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    summaries, outputs = {}, []
    for benchmark in prepared["benchmarks"]:
        name = benchmark["id"]
        rows = selected_rows(prepared, benchmark, args.partition, args.limit)
        budget = next(
            item["max_output_tokens"] for item in protocol["datasets"] if item["id"] == name
        )
        task = None if name == "humanevalplus" else frozen_task(name, rows)
        docs = rows if task is None else list(task.eval_docs)
        values = {}
        for doc in docs:
            started = time.monotonic()
            prompt = doc["prompt"] if task is None else task.doc_to_text(doc)
            if args.chat:
                prompt = model.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            if len(model.tok_encode(prompt)) + budget > 4096:
                raise ValueError(
                    "prompt plus output budget exceeds context; refusing silent truncation"
                )
            if budget:
                request = Instance(
                    request_type="generate_until",
                    doc=doc,
                    idx=0,
                    arguments=(
                        prompt,
                        {
                            "until": [],
                            "max_gen_toks": budget,
                            "do_sample": False,
                            "temperature": 0.0,
                        },
                    ),
                )
                response = model.generate_until([request])[0]
                if task is None:
                    code = sanitize(
                        response if args.chat else doc["prompt"] + response,
                        entrypoint=doc["entry_point"],
                    )
                    execution = run_python(code, doc["test"], doc["entry_point"])
                    metrics = {"compiled_plus_pass@1": execution["status"] == "pass"}
                else:
                    execution = None
                    metrics = score_text(task, doc, response)
            else:
                choices = task.doc_to_choice(doc)
                requests = [
                    Instance(
                        request_type="loglikelihood",
                        doc=doc,
                        idx=i,
                        arguments=(prompt, task.config.target_delimiter + choice),
                    )
                    for i, choice in enumerate(choices)
                ]
                response = model.loglikelihood(requests)
                metrics = task.process_results(doc, response)
                execution = None
            row = {
                "benchmark": name,
                "task_id": doc["speck_task_id"],
                "prompt": prompt,
                "response": response,
                "metrics": metrics,
                "execution": execution,
                "wall_seconds": time.monotonic() - started,
                "returned_text_tokens": len(model.tok_encode(response)) if budget else 0,
            }
            outputs.append(row)
            with (Path(args.output) / "outputs.jsonl").open("a") as handle:
                handle.write(json.dumps(row, default=_json_scalar) + "\n")
            for key, value in metrics.items():
                values.setdefault(key, []).append(value)
        aggregates = {}
        for key, items in values.items():
            aggregate = (
                (lambda x: sum(x) / len(x))
                if task is None
                else task.aggregation()[key.split(",")[0]]
            )
            aggregates[key] = aggregate(items)
        summaries[name] = {"tasks": len(docs), "metrics": aggregates, "max_output_tokens": budget}
    return {
        "status": "pass",
        "scorers": scorers,
        "model": args.model,
        "revision": args.revision,
        "partition": args.partition,
        "limit": args.limit,
        "chat": args.chat,
        "enable_thinking": False,
        "temperature": 0,
        "samples": 1,
        "device": args.device,
        "dtype": "bfloat16",
        "max_context_tokens": 4096,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "numpy",
                "datasets",
                "langdetect",
                "immutabledict",
            )
        },
        "results": summaries,
        "boundary": "Custom frozen subsets; limited runs are pipeline checks, not model rankings. Code uses the frozen HF compiled plus tests with a 15-second wall timeout.",
    }


def _json_scalar(value):
    return value.item()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol")
    parser.add_argument("prepared")
    parser.add_argument("--output", required=True)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--partition", choices=("development", "final"), default="development")
    parser.add_argument(
        "--limit", type=int, default=8, help="0 evaluates the complete chosen partition"
    )
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    check_sandbox()
    protocol, prepared = (
        json.loads(Path(args.protocol).read_text()),
        json.loads(Path(args.prepared).read_text()),
    )
    if file_sha256(args.protocol) != prepared["protocol_sha256"]:
        raise ValueError("prepared protocol identity mismatch")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "status": "running",
        "protocol_sha256": file_sha256(args.protocol),
        "prepared_sha256": file_sha256(args.prepared),
        "implementation": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("capability.py", "code_runner.py", "tools.py")
        },
    }
    atomic_json(output / "result.json", result)
    try:
        result.update(
            qualify(prepared, protocol) if args.qualify else evaluate(args, prepared, protocol)
        )
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["outputs_sha256"] = (
            file_sha256(output / "outputs.jsonl") if (output / "outputs.jsonl").exists() else None
        )
        atomic_json(output / "result.json", json.loads(json.dumps(result, default=_json_scalar)))
    print(json.dumps(result, indent=2, default=_json_scalar))


if __name__ == "__main__":
    main()

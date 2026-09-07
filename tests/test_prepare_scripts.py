from scripts import (
    code_contamination_scan,
    data_prepare,
    sft_prepare,
    stack_edu_sample,
    stack_v3_expand,
    stack_v3_qualify,
    stack_v3_refine,
    tokenizer_evaluate,
    tokenizer_prepare,
    tokenizer_sample_prepare,
    tokenizer_train,
)


def test_prepare_script_argument_parsers_are_import_safe():
    tokenizer = tokenizer_prepare.parse_args([])
    contamination = code_contamination_scan.parse_args(["code.json", "--restart"])
    stack_edu = stack_edu_sample.parse_args(["stack-edu.json", "--restart"])
    data = data_prepare.parse_args(["custom-data", "--restart"])
    sft = sft_prepare.parse_args(["custom-sft", "--restart"])

    assert tokenizer.experiment == "experiments/Speck1-140M"
    assert contamination.config == "code.json" and contamination.restart
    assert stack_edu.config == "stack-edu.json" and stack_edu.restart
    assert data.experiment == "custom-data" and data.restart
    assert sft.experiment == "custom-sft" and sft.restart

    sample = tokenizer_sample_prepare.parse_args(["tokenizer.json", "--restart"])
    train = tokenizer_train.parse_args(
        ["tokenizer.json", "--candidate", "speck-32k", "--prepare-baselines"]
    )
    evaluate = tokenizer_evaluate.parse_args(["tokenizer.json"])
    stack = stack_v3_qualify.parse_args(["stack-v3.json", "--restart"])
    expand = stack_v3_expand.parse_args(["stack-v3-v1b.json", "--restart"])
    refine = stack_v3_refine.parse_args(["stack-v3-v2.json", "--restart"])
    assert sample.config == train.config == evaluate.config == "tokenizer.json"
    assert sample.restart and train.candidate == ["speck-32k"] and train.prepare_baselines
    assert stack.config == "stack-v3.json" and stack.restart
    assert expand.config == "stack-v3-v1b.json" and expand.restart
    assert refine.config == "stack-v3-v2.json" and refine.restart

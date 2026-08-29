from scripts import data_prepare, sft_prepare, tokenizer_prepare


def test_prepare_script_argument_parsers_are_import_safe():
    tokenizer = tokenizer_prepare.parse_args([])
    data = data_prepare.parse_args(["custom-data", "--restart"])
    sft = sft_prepare.parse_args(["custom-sft", "--restart"])

    assert tokenizer.experiment == "experiments/Speck1-140M"
    assert data.experiment == "custom-data" and data.restart
    assert sft.experiment == "custom-sft" and sft.restart

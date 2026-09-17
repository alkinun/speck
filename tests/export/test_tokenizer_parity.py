import io
import json
import shutil

import pytest
import sentencepiece

from speck.export.transformers import CURRENT_TOKENIZATION_SOURCE, validate_tokenizer_parity
from speck.tokenization.chat import ChatTokenizer
from speck.tokenization.tokenizer import Tokenizer


@pytest.mark.parametrize("version", [1, 2])
def test_exported_chat_uses_native_ids_and_rejects_template_drift(tmp_path, version):
    pytest.importorskip("transformers")
    model = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(["Hello world! A model reads text.\nSecond line."] * 10),
        model_writer=model,
        vocab_size=32,
        model_type="bpe",
        hard_vocab_limit=False,
        num_threads=1,
        shuffle_input_sentence=False,
        minloglevel=2,
    )
    source = tmp_path / "source.model"
    source.write_bytes(model.getvalue())
    tokenizer = ChatTokenizer(Tokenizer(source), version)
    output = tmp_path / "export"
    tokenizer.save_pretrained(output)
    config_path = output / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config.update(
        auto_map={"AutoTokenizer": ["tokenization_speck.SpeckTokenizer", None]},
        tokenizer_class="SpeckTokenizer",
    )
    config_path.write_text(json.dumps(config))
    shutil.copy2(CURRENT_TOKENIZATION_SOURCE, output / "tokenization_speck.py")
    metadata = {"training_phase": "sft", "resolved": {"tokenizer": tokenizer.metadata()}}

    result = validate_tokenizer_parity(output, metadata)
    assert result["passed"] and result["chat_cases"] == 2
    (output / "chat_template.jinja").write_text("wrong template")
    with pytest.raises(ValueError, match="template differs"):
        validate_tokenizer_parity(output, metadata)

"""Exact SentencePiece backend for base and assistant Transformers exports.

Retained from specklabs/Speck1-140M-Instruct at
16ad80599d499490b70317770a84a18466719bba, now bundled and checked with current code.
"""

from typing import ClassVar

from transformers import SentencePieceBackend


class SpeckTokenizer(SentencePieceBackend):
    model_input_names: ClassVar[list[str]] = ["input_ids", "attention_mask"]
    padding_side = "right"

    def __init__(self, add_bos_token=True, add_eos_token=False, **kwargs):
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        super().__init__(**kwargs)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos = [self.bos_token_id] if self.add_bos_token else []
        eos = [self.eos_token_id] if self.add_eos_token else []
        first = bos + token_ids_0 + eos
        if token_ids_1 is None:
            return first
        return first + bos + token_ids_1 + eos

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        first_length = len(token_ids_0) + self.add_bos_token + self.add_eos_token
        if token_ids_1 is None:
            return [0] * first_length
        second_length = len(token_ids_1) + self.add_bos_token + self.add_eos_token
        return [0] * first_length + [1] * second_length

    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1=None, already_has_special_tokens=False
    ):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0, token_ids_1, already_has_special_tokens=True
            )
        bos = [1] if self.add_bos_token else []
        eos = [1] if self.add_eos_token else []
        mask = bos + [0] * len(token_ids_0) + eos
        if token_ids_1 is not None:
            mask += bos + [0] * len(token_ids_1) + eos
        return mask

    def _decode(self, token_ids, skip_special_tokens=False, **kwargs):
        del kwargs
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [token for token in token_ids if token not in self.all_special_ids]
            return self.sp_model.decode(token_ids)
        output, ordinary = [], []
        for token in token_ids:
            if token < self.sp_model.vocab_size():
                ordinary.append(token)
                continue
            if ordinary:
                output.append(self.sp_model.decode(ordinary))
                ordinary = []
            output.append(self.convert_ids_to_tokens(token))
        if ordinary:
            output.append(self.sp_model.decode(ordinary))
        return "".join(output)

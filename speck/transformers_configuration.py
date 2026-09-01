"""Transformers configuration shipped with exported Speck checkpoints."""

from dataclasses import fields

from transformers import PreTrainedConfig

from .architecture_speck import ArchitectureConfig


class SpeckConfig(PreTrainedConfig):
    model_type = "speck"
    has_no_defaults_at_init = True

    def __init__(self, **values):
        architecture_fields = {field.name for field in fields(ArchitectureConfig)}
        architecture = {key: values.pop(key) for key in tuple(values) if key in architecture_fields}
        parsed = ArchitectureConfig.from_dict(architecture)
        for key, value in parsed.export().items():
            setattr(self, key, value)
        use_cache = values.pop("use_cache", True)
        for key in ("is_decoder", "is_encoder_decoder", "tie_word_embeddings"):
            values.pop(key, None)
        super().__init__(
            bos_token_id=parsed.bos_token_id,
            eos_token_id=parsed.eos_token_id,
            is_decoder=True,
            is_encoder_decoder=False,
            tie_word_embeddings=True,
            use_cache=use_cache,
            **values,
        )

    def architecture_config(self):
        allowed = {field.name for field in fields(ArchitectureConfig)}
        return ArchitectureConfig.from_dict(
            {key: getattr(self, key) for key in allowed if hasattr(self, key)}
        )


__all__ = ["SpeckConfig"]

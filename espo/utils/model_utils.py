# Adapted based on huggingface/open-r1: https://github.com/huggingface/open-r1/blob/6a0cd5c8ad031fc75118a4ce7f42a4860c3d8dea/src/open_r1/utils/model_utils.py


import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from ..configs import GRPOConfig


def get_tokenizer(model_args: ModelConfig, training_args: GRPOConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer

def dummy_prepare_inputs_for_generation(self, input_ids=None, **kwargs):
    # 如果你的 DreamModel 没有 generate 逻辑，可以直接原样返回
    return {"input_ids": input_ids, **kwargs}



def get_model(model_args: ModelConfig, training_args:GRPOConfig) -> AutoModel:
    """Get the model"""
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False ,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    if not hasattr(model, "prepare_inputs_for_generation"):
        print("Patching prepare_inputs_for_generation to model...")
        import types
        model.prepare_inputs_for_generation = types.MethodType(dummy_prepare_inputs_for_generation, model)

    return model

# Vendored verbatim from SkyworkAI/skywork-o1-prm-inference/model_utils/prm_model.py (Apache 2.0).
# Copyright 2022 The HuggingFace Team.
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .modeling_base import PreTrainedModelWrapper


class ValueHead(nn.Module):
    """Per-token scalar reward head — Linear(hidden, 1)."""

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob
        self.dropout = (
            nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        )

        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        self.summary = nn.Linear(hidden_size, 1)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        return self.summary(output)


class PRM_MODEL(PreTrainedModelWrapper):
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)
        if not any(hasattr(self.pretrained_model, a) for a in self.lm_head_namings):
            raise ValueError("The model does not have a language model head.")
        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)
        self._init_weights(**v_head_kwargs)

    def _init_weights(self, **kwargs):
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy == "normal":
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_past_key_values=False,
        return_probs=False,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        if (
            getattr(self, "is_peft_model", False)
            and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING"
        ):
            kwargs.pop("past_key_values")

        base_model_output = self.pretrained_model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        value = self.v_head(last_hidden_state).squeeze(-1)  # [B, T]

        if return_probs:
            value = torch.nn.functional.sigmoid(value)

        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        if return_past_key_values:
            return (lm_logits, loss, value, base_model_output.past_key_values)
        return (lm_logits, loss, value)

    def generate(self, *args, **kwargs):
        return self.pretrained_model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        if not getattr(self, "is_peft_model", False):
            sd = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            sd = {}
        v_sd = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_sd.items():
            sd[f"v_head.{k}"] = v
        return sd

    def post_init(self, state_dict):
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            dm = self.pretrained_model.hf_device_map.values()
            if "cpu" in dm or "disk" in dm:
                raise ValueError("CPU/disk offloading is not supported for ValueHead models.")
            first = list(set(dm))[0]
            if isinstance(first, int):
                first = f"cuda:{first}"
            self.v_head = self.v_head.to(first)

            def _hook(_module, _input, outputs):
                new = ()
                for o in outputs:
                    new += (o.to(first) if isinstance(o, torch.Tensor) else o,)
                return new

            self.register_forward_hook(_hook)
            self.is_sequential_parallel = True

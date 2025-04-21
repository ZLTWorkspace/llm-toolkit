from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from torch import nn

from peft.config import PeftConfig, _check_and_remove_unused_kwargs, MIN_EXPECTED_CONFIG_KEYS


@dataclass
class SQALoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`SQALoraModel`].

    Args:
    sparsity_ratio (`float`, *optional*, defaults to 0.5):
        Sparse ratio of base model. For example, 0.5 indicates half of the parameters in a linear is 0.
    sparse_preserve_accuracy (`bool`, *optional*, defaults to `False`):
        Merge sparse W into A and B to preserve accuracy. Default is `False`.
    sparse_prune_largest (`bool`, *optional*, defaults to `False`):
        If True, prune the largest weights, otherwise prune the smallest weights. Default is `False`.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D "
                "(if the model is a PreTrainedModel, the output layer excluded)."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )

    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                "When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a>"
                " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
                " was proven to work better. Otherwise, it will use the original default"
                " value of `lora_alpha/r`."
            )
        },
    )
    init_lora_weights: bool | Literal["gaussian"] = field(
        default=True,
        metadata={
            "help": (
                "How to initialize the weights of the LoRA layers. "
                "Passing True (default) results in the default initialization from the reference implementation from "
                "Microsoft, with the LoRA B weight being set to 0. This means that without further training, the LoRA "
                "adapter will be a no-op. "
                "Setting the initialization to False leads to random initialization of LoRA A and B, meaning that LoRA "
                "is not a no-op before training; this setting is intended for debugging purposes. "
                "Passing `'gaussian'` results in Gaussian initialization scaled by the LoRA rank for linear and layers. "
            ),
        },
    )
    lora_bias: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable the bias term for the LoRA B parameter. Typically, this should be disabled. The "
                "main use case for this is when the LoRA weights were extracted from fully fine-tuned parameters so "
                "the bias of those parameters can be taken into account."
            )
        },
    )
    sparse_preserve_mode: int = field(
        default=0, metadata={"help": "Merge sparse W into A and B to preserve accuracy. Default is 0."}
    )

    def __post_init__(self):
        super().__post_init__()
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

    @classmethod
    def from_peft_type(cls, **kwargs):
        r"""
        This method loads the configuration of your adapter model from a set of kwargs.

        The appropriate configuration type is determined by the `peft_type` argument. If `peft_type` is not provided,
        the calling class type is instantiated.

        Args:
            kwargs (configuration keyword arguments):
                Keyword arguments passed along to the configuration initialization.
        """
        # Avoid circular dependency .. TODO: fix this with a larger refactor
        from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        # TODO: this hack is needed to fix the following issue (on commit 702f937):
        # if someone saves a default config and loads it back with `PeftConfig` class it yields to
        # not loading the correct config class.
        #
        # from peft import AdaLoraConfig, PeftConfig
        # peft_config = AdaLoraConfig()
        # print(peft_config)
        # >>> AdaLoraConfig(peft_type=<PeftType.ADALORA: 'ADALORA'>, auto_mapping=None, base_model_name_or_path=None,
        # revision=None, task_type=None, inference_mode=False, r=8, target_modules=None, lora_alpha=8, lora_dropout=0.0, ...
        #
        # peft_config.save_pretrained("./test_config")
        # peft_config = PeftConfig.from_pretrained("./test_config")
        # print(peft_config)
        # >>> PeftConfig(peft_type='ADALORA', auto_mapping=None, base_model_name_or_path=None, revision=None, task_type=None, inference_mode=False)

        if "peft_type" in kwargs and kwargs["peft_type"] is not None:
            peft_type = kwargs["peft_type"]
            config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]
        else:
            config_cls = cls

        try:
            config = config_cls(**kwargs)
        except TypeError as exc:
            # Here we potentially handle forward compatibility. Sometimes new keywords are added to configs, which makes
            # new configs incompatible with older PEFT versions. We catch these and remove them to allow the program to
            # continue, but warn the user about it.

            # First check if the error is due to unexpected keyword arguments, we don't want to accidentally catch
            # other TypeErrors.
            if "got an unexpected keyword argument" not in str(exc):
                raise exc

            filtered_kwargs, unexpected_kwargs = _check_and_remove_unused_kwargs(config_cls, kwargs)
            if not MIN_EXPECTED_CONFIG_KEYS.issubset(set(filtered_kwargs.keys())):
                raise TypeError(
                    f"The {cls.__name__} config that is trying to be loaded is missing required keys: "
                    f"{MIN_EXPECTED_CONFIG_KEYS}."
                )

            warnings.warn(
                f"Unexpected keyword arguments {sorted(unexpected_kwargs)} for class {config_cls.__name__}, these are "
                "ignored. This probably means that you're loading a configuration file that was saved using a "
                "higher version of the library and additional parameters have been introduced since. It is "
                "highly recommended to upgrade the PEFT version before continuing (e.g. by running `pip install "
                "-U peft`)."
            )
            config = config_cls.from_peft_type(**filtered_kwargs)
        return config
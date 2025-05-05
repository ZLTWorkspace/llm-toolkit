from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Union


@dataclass
class SQALoraConfig:
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
    quant_method: str = field(
        default="nf4", metadata={"help": "Quantization method to use, nf4 -> bnb nf4, mxfp4 -> xx."}
    )

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the configuration as a JSON file called *sqalora_config.json*
        in `save_directory`.

        Raises
        ------
        ValueError
            If `save_directory` points to an existing file.
        """
        if os.path.isfile(save_directory):
            raise ValueError(
                f"`save_directory` ({save_directory}) is a file. "
                "It must be a directory."
            )
        os.makedirs(save_directory, exist_ok=True)

        cfg_dict = dataclasses.asdict(self)

        if isinstance(self.target_modules, set):
            cfg_dict["target_modules"] = sorted(self.target_modules)

        file_name = os.path.join(save_directory, "sqalora_config.json")
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2, ensure_ascii=False, sort_keys=True)


    @classmethod
    def from_pretrained(cls, load_directory: str) -> SQALoraConfig:
        """
        Load a configuration from *sqalora_config.json* located in
        `load_directory` (or load_directory itself can be the full path
        to the JSON file).

        Returns
        -------
        SQALoraConfig
        """
        if os.path.isdir(load_directory):
            file_name = os.path.join(load_directory, "sqalora_config.json")
        else:
            file_name = load_directory

        if not os.path.isfile(file_name):
            raise FileNotFoundError(
                f"Could not find sqalora configuration at `{file_name}`."
            )

        with open(file_name, encoding="utf-8") as f:
            loaded_dict = json.load(f)

        return cls(**loaded_dict)

    def __post_init__(self):
        if isinstance(self.target_modules, list):
            self.target_modules = set(self.target_modules)

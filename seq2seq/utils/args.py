from typing import Optional
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class PeftArguments:

    """
    Arguments with PEFT hyperparameters
    """

    peft_weights : str = field(
        default=None,
        metadata={"help": "Path to the weights file for PEFT"}
    )
    peft_config : str = field(
        default="",
        metadata={"help": "Path to the config file for PEFT"}
    )
    task_type : str = field(
        default="TaskType.CAUSAL_LM",
        metadata={"help": "Task type for PEFT"}
    )
    lora_r : float = field(
        default=0.0,
        metadata={"help": "LoRA regularization term"}
    )
    lora_alpha : float = field(
        default=0.0,
        metadata={"help": "LoRA alpha term"}
    )
    lora_dropout : float = field(
        default=0.0,
        metadata={"help": "LoRA dropout term"}
    )

@dataclass
class HuggingFaceArguments:

    """
    Arguments with HuggingFace hyperparameters
    """

    hf_key : str = field(
        default="",
        metadata={"help": "HuggingFace API key"}
    )
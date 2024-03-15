# Code from taprosoft's github
from dataclasses import dataclass, field
import transformers
import torch
import copy
from typing import Dict, Sequence
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        #sources = [f"{self.tokenizer.bos_token}{example['input_ids']}" for example in instances]
        #targets = [f"{example['labels']}{self.tokenizer.eos_token}" for example in instances]
        
        sources = []
        targets = []
        max_length = 0

        for example in instances:

            sources.append(example['input_ids'])

            target_temp = example['labels'] + [self.tokenizer.eos_token_id]
            if(target_temp[0] == self.tokenizer.bos_token_id):
                target_temp = target_temp[1:]
            targets.append(target_temp)

            max_length = max(max_length, len(example['input_ids']) + len(target_temp))

        #sources = [example['input_ids'] for example in instances]
        #targets = [example['labels'] + [self.tokenizer.eos_token_id] for example in instances]
        #max_length = max(len(s) for s in sources) if self.predict_with_generate else max(len(s) + len(t) for s, t in zip(sources, targets))

        # Tokenize
        #tokenized_sources_with_prompt = self.tokenizer(
        #    sources,
        #    max_length=self.source_max_len,
        #    truncation=True,
        #    add_special_tokens=False,
        #)
        #tokenized_targets = self.tokenizer(
        #    targets,
        #    max_length=self.target_max_len,
        #    truncation=True,
        #    add_special_tokens=False,
        #)
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        masks = []
        for tokenized_source, tokenized_target in zip(
            #tokenized_sources_with_prompt['input_ids'],
            #tokenized_targets['input_ids']
            sources,targets
        ):
            if not self.predict_with_generate:
                remainder = [self.tokenizer.pad_token_id] * (max_length - len(tokenized_source + tokenized_target))
                if self.tokenizer.padding_side == "right":
                    input = tokenized_source + tokenized_target + remainder
                else:
                    input = remainder + tokenized_source + tokenized_target
                input_ids.append(torch.tensor(input))

                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(max_length - len(tokenized_target))] + copy.deepcopy(tokenized_target))
                    )
                    masks.append(
                        torch.tensor([0 for _ in range(max_length - len(tokenized_target))] + [1 for _ in range(len(tokenized_target))])
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(input)))
                    masks.append(torch.tensor([1 for _ in range(len(input))]))
            else:
                remainder = [self.tokenizer.pad_token_id] * (max_length - len(tokenized_source))
                if self.tokenizer.padding_side == "right":
                    input = tokenized_source + remainder
                else:
                    input = remainder + tokenized_source
                input_ids.append(torch.tensor(input))
                masks.append(torch.tensor([1 for _ in range(len(input))]))
        # Apply padding
        #input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        #labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': torch.stack(input_ids),
            #'attention_mask': torch.stack(masks),
            'attention_mask':torch.stack(input_ids).ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = torch.stack(labels)
        return data_dict
    

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        labels = [feature["labels"] for feature in instances] if "labels" in instances[0].keys() else None
        input_ids = [feature["input_ids"] for feature in instances] if "input_ids" in instances[0].keys() else None
        #input_ids, labels = tuple(
        #    [instance[key] for instance in instances] for key in ("input_ids", "labels")
        #)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(
                self.tokenizer.pad_token_id
            ),  # HF-Transformers Attention-Masking https://huggingface.co/docs/transformers/glossary#a
        )
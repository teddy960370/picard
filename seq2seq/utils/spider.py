import json
import numpy as np
from typing import Optional
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from seq2seq.utils.dataset import DataTrainingArguments, normalize, serialize_schema
from seq2seq.utils.trainer import Seq2SeqTrainer, EvalPrediction,Trainer


def spider_get_input(
    question: str,
    serialized_schema: str,
    prefix: str,
) -> str:
    
    if prefix is None:
        return question.strip() + " " + serialized_schema.strip()
    else:

        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        B_INST, E_INST = "[INST]", "[/INST]"

        user_prompt = f"""### Input: 
{question.strip()} 
### Context: 
{serialized_schema.strip()}
"""

        # Chat model prompt
        prompt = f"{B_INST} {B_SYS}{prefix.strip()}{E_SYS} {E_INST}{user_prompt.strip()}\n"

        return prompt


def spider_get_target(
    query: str,
    db_id: str,
    normalize_query: bool,
    target_with_db_id: bool,
) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)
    #return f"{db_id} | {_normalize(query)}" if target_with_db_id else _normalize(query)

    if target_with_db_id:
        return f"### Response: {db_id} | {_normalize(query)}"
    else:
        return "### Response: " +_normalize(query)


def spider_add_serialized_schema(ex: dict, data_training_args: DataTrainingArguments) -> dict:
    serialized_schema = serialize_schema(
        question=ex["question"],
        db_path=ex["db_path"],
        db_id=ex["db_id"],
        db_column_names=ex["db_column_names"],
        db_table_names=ex["db_table_names"],
        schema_serialization_type=data_training_args.schema_serialization_type,
        schema_serialization_randomized=data_training_args.schema_serialization_randomized,
        schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
        schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
        normalize_query=data_training_args.normalize_query,
    )
    return {"serialized_schema": serialized_schema}


def spider_pre_process_function(
    batch: dict,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    data_training_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    prefix = data_training_args.source_prefix if data_training_args.source_prefix is not None else ""

    inputs = [
        spider_get_input(question=question, serialized_schema=serialized_schema, prefix=prefix)
        for question, serialized_schema in zip(batch["question"], batch["serialized_schema"])
    ]

    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        return_overflowing_tokens=False,
    )

    targets = [
        spider_get_target(
            query=query,
            db_id=db_id,
            normalize_query=data_training_args.normalize_query,
            target_with_db_id=data_training_args.target_with_db_id,
        )
        for db_id, query in zip(batch["db_id"], batch["query"])
    ]

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class SpiderTrainer(Trainer):
    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)
        #label_ids = [f["labels"] for f in features]
        temp_label_ids = [f["labels"] for f in features]

        # list reshape to numpy array
        label_ids = np.zeros([len(temp_label_ids),len(max(temp_label_ids,key = lambda x: len(x)))], dtype=int)
        for i,j in enumerate(temp_label_ids):
            label_ids[i][0:len(j)] = j

        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            _label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
            _predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)

        decoded_label_ids = self.tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
        metas = [
            {
                "query": x["query"],
                "question": x["question"],
                "context": context,
                "label": label,
                "db_id": x["db_id"],
                "db_path": x["db_path"],
                "db_table_names": x["db_table_names"],
                "db_column_names": x["db_column_names"],
                "db_foreign_keys": x["db_foreign_keys"],
            }
            for x, context, label in zip(examples, inputs, decoded_label_ids)
        ]
        predictions = self.tokenizer.batch_decode(_predictions, skip_special_tokens=True)
        assert len(metas) == len(predictions)

        buffer = []
        for pred in predictions:
            if pred.rfind("### Response: ") != -1:
                pos = pred.rfind("### Response: ") + len("### Response: ")
                buffer.append(pred[pos:])
            else:
                buffer.append(pred)
        predictions = buffer

        with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
            json.dump(
                [dict(**{"prediction": prediction}, **meta) for prediction, meta in zip(predictions, metas)],
                f,
                indent=4,
            )
        return EvalPrediction(inputs = inputs ,predictions=predictions, label_ids=label_ids, metas=metas)

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        inputs , predictions, label_ids, metas = eval_prediction
        # Remove prediction prefix which is prompt
        #predictions = [prediction[len(input):] for input , prediction in zip(inputs,predictions)]
        # buffer = []
        # for pred in predictions:
        #     if pred.rfind("### Response: ") != -1:
        #         pos = pred.rfind("### Response: ") + len("### Response: ")
        #         buffer.append(pred[pos:])
        #     else:
        #         buffer.append(pred)
        # predictions = buffer

        if self.target_with_db_id:
            # Remove database id from all predictions
            predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]
        # TODO: using the decoded reference labels causes a crash in the spider evaluator
        # if self.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        # decoded_references = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # references = [{**{"query": r}, **m} for r, m in zip(decoded_references, metas)]
        references = metas
        return self.metric.compute(predictions=predictions, references=references)

# Set up logging
import sys
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

import os
import json
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict, fields
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.models.auto import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
from transformers.data.data_collator import DataCollatorForSeq2Seq,DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint, set_seed
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tokenizers import AddedToken
from seq2seq.utils.args import ModelArguments,PeftArguments,HuggingFaceArguments
from seq2seq.utils.picard_model_wrapper import PicardArguments, PicardLauncher, with_picard
from seq2seq.utils.dataset import DataTrainingArguments, DataArguments
from seq2seq.utils.dataset_loader import load_dataset
from seq2seq.utils.spider import SpiderTrainer
from seq2seq.utils.cosql import CoSQLTrainer
from huggingface_hub import login

import torch
from transformers import BitsAndBytesConfig,Seq2SeqTrainer,TextStreamer
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType,prepare_model_for_kbit_training
from DataCollatorForCausalLM import DataCollatorForCausalLM,DataCollatorForSupervisedDataset


def check_baseline(tokenizer,model):

    eval_prompt = """[INST]<<SYS>>
    You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.
    You must output the SQL query that answers the question.
    <</SYS>>[/INST]
    
    ### Input:
    Which Class has a Frequency MHz larger than 91.5, and a City of license of hyannis, nebraska?

    ### Context:
    CREATE TABLE table_name_12 (class VARCHAR, frequency_mhz VARCHAR, city_of_license VARCHAR)

    ### Response:
    
    """
    # {'question': 'Name the comptroller for office of prohibition', 'context': 'CREATE TABLE table_22607062_1 (comptroller VARCHAR, ticket___office VARCHAR)', 'answer': 'SELECT comptroller FROM table_22607062_1 WHERE ticket___office = "Prohibition"'}
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    model.eval()
    with torch.no_grad():

        streamer = TextStreamer(tokenizer)

        _ = model.generate(**model_input, streamer=streamer, max_new_tokens=50, temperature=0.01)
        print("End of generation")
        #generate_ids = model.generate(**model_input, max_new_tokens=100)
        #except_prompts = generate_ids[:, model_input.input_ids.shape[1]:]
        #print(tokenizer.decode(except_prompts[0], skip_special_tokens=True))

def main() -> None:
    # See all possible arguments by passing the --help flag to this script.
    parser = HfArgumentParser(
        (PicardArguments, ModelArguments, DataArguments, DataTrainingArguments, Seq2SeqTrainingArguments,PeftArguments,HuggingFaceArguments)
    )
    picard_args: PicardArguments
    model_args: ModelArguments
    data_args: DataArguments
    data_training_args: DataTrainingArguments
    training_args: Seq2SeqTrainingArguments
    peft_args: PeftArguments
    huggingface_args: HuggingFaceArguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        picard_args, model_args, data_args, data_training_args, training_args,peft_args,huggingface_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 3 and sys.argv[1].startswith("--local_rank") and sys.argv[2].endswith(".json"):
        data = json.loads(Path(os.path.abspath(sys.argv[2])).read_text())
        data.update({"local_rank": int(sys.argv[1].split("=")[1])})
        picard_args, model_args, data_args, data_training_args, training_args,peft_args,huggingface_args = parser.parse_dict(args=data)
    else:
        picard_args, model_args, data_args, data_training_args, training_args,peft_args,huggingface_args = parser.parse_args_into_dataclasses()
    
    #login Huggingface
    if huggingface_args.hf_key is not None:
        login(token = huggingface_args.hf_key)

    training_args.do_train = False
    training_args.do_eval = True


    # If model_name_or_path includes ??? instead of the number of steps, 
    # we load the latest checkpoint.
    if 'checkpoint-???' in model_args.model_name_or_path:
        model_args.model_name_or_path = get_last_checkpoint(
            os.path.dirname(model_args.model_name_or_path))
        logger.info(f"Resolve model_name_or_path to {model_args.model_name_or_path}")

    combined_args_dict = {
        **asdict(picard_args),
        **asdict(model_args),
        **asdict(data_args),
        **asdict(data_training_args),
        **training_args.to_sanitized_dict(),
    }
    combined_args_dict.pop("local_rank", None)


    # Initialize config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        max_length=data_training_args.max_target_length,
        num_beams=data_training_args.num_beams,
        num_beam_groups=data_training_args.num_beam_groups,
        diversity_penalty=data_training_args.diversity_penalty,
        gradient_checkpointing=training_args.gradient_checkpointing,
        use_cache=not training_args.gradient_checkpointing,
    )

    

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        #add_eos_token=True
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Only fast tokenizers are currently supported"
    if isinstance(tokenizer, T5TokenizerFast):
        # In T5 `<` is OOV, see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/restore_oov.py
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    # Check if the pad token is already in the tokenizer vocabulary
    if '<pad>' not in tokenizer.get_vocab():
        # Add the pad token
        #tokenizer.pad_token_id = 0
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
    
    #if tokenizer.pad_token is None:
    #    smart_tokenizer_and_embedding_resize(
    #        special_tokens_dict=dict(pad_token="[PAD]"),
    #        tokenizer=tokenizer,
    #        model=model,
    #    )


    #Check if the mask token is already in the tokenizer vocabulary
    if '<mask>' not in tokenizer.get_vocab():
        # Add the mask token
        tokenizer.add_special_tokens({"mask_token":"<mask>"})
        
    #tokenizer.padding_side = 'left'

    # Load dataset
    metric, dataset_splits = load_dataset(
        data_args=data_args,
        model_args=model_args,
        data_training_args=data_training_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )

    # Initialize Picard if necessary
    with PicardLauncher() if picard_args.launch_picard and training_args.local_rank <= 0 else nullcontext(None):
        # Get Picard model class wrapper
        if picard_args.use_picard:
            model_cls_wrapper = lambda model_cls: with_picard(
                model_cls=model_cls, picard_args=picard_args, tokenizer=tokenizer, schemas=dataset_splits.schemas
            )
        else:
            model_cls_wrapper = lambda model_cls: model_cls

        # Initialize model
        model = model_cls_wrapper(AutoModelForCausalLM).from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            quantization_config=bnb_config,
        )
        if isinstance(model, T5ForConditionalGeneration):
            model.resize_token_embeddings(len(tokenizer))

        #Resize the embeddings
        model.resize_token_embeddings(len(tokenizer))

        #Configure the pad token in the model
        model.config.pad_token_id = tokenizer.pad_token_id

        #Configure the mask token in the model
        model.config.mask_token_id = tokenizer.mask_token_id

        # Check if they are equal
        assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"

        #model.config.use_cache = False
        
        if peft_args.peft_weights is not None:
            model = PeftModel.from_pretrained(
                model,
                peft_args.peft_weights,
                torch_dtype=torch.float16,
                is_trainable=False,
            )
            model.config.use_cache = True
        else:
            peft_config = LoraConfig(
                task_type = TaskType.CAUSAL_LM, 
                inference_mode = False, 
                r = peft_args.lora_r, 
                lora_alpha = peft_args.lora_alpha, 
                lora_dropout = peft_args.lora_dropout
                #r = 256,
                #lora_alpha = 512,
            )
            model = get_peft_model(model, peft_config)
            model.config.use_cache = False
            
            model.print_trainable_parameters()


        model.enable_input_require_grads()
        #model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)


        if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
            logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )

        # Initialize Trainer
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "metric": metric,
            "train_dataset": dataset_splits.train_split.dataset if training_args.do_train else None,
            "eval_dataset": dataset_splits.eval_split.dataset if training_args.do_eval else None,
            "eval_examples": dataset_splits.eval_split.examples if training_args.do_eval else None,
            "tokenizer": tokenizer,
            "data_collator" : DataCollatorForCausalLM(
                tokenizer=tokenizer,
                source_max_len=data_training_args.max_target_length,
                target_max_len=data_training_args.max_target_length,
                train_on_source=False,
                predict_with_generate=False,
            ),
            "ignore_pad_token_for_loss": data_training_args.ignore_pad_token_for_loss,
            "target_with_db_id": data_training_args.target_with_db_id,
        }
        #using spidertrainer as it is.
        if data_args.dataset in ["spider", "spider_realistic", "spider_syn", "spider_dk"]:
            trainer = SpiderTrainer(**trainer_kwargs)
        elif data_args.dataset in ["cosql", "cosql+spider"]:
            trainer = CoSQLTrainer(**trainer_kwargs)
        else:
            raise NotImplementedError()

        check_baseline(tokenizer,model)
        #check_baseline(tokenizer,model)





if __name__ == "__main__":

    main()
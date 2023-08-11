'''
I thought there needed to be a ground up design of the feedback system.
Tried to finesse it with run.py but lowkey need ot build it ground up too many differences tbh
'''
import argparse
import time
import tasks
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, HfArgumentParser, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification
from typing import Union, Optional
import torch
from torch.nn.parameter import Parameter
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
from tqdm import tqdm
from tasks import get_task
import json
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from metrics import calculate_metric
from utils import *
from trainer import OurTrainer
import random

@dataclass
class OurArguments(TrainingArguments):
    '''
    Our custom arguments for the trainer
    '''
    task_name: str = "WIC"
    # number of examples

     # Model loading
    model_name: str = "meta-llama/Llama-2-7b-hf" # HuggingFace model name
    load_float16: bool = True # load model parameters as float16
    load_bfloat16: bool = False # load model parameters as bfloat16
    load_int8: bool = False # load model parameters as int8
    max_length: int = 2048 # max length the model can take
    no_auto_device: bool = False # do not load model by auto device; should turn this on when using FSDP
    #model loading -> need ot offlad this sumewhere
    trainer: str = "zo"
    # calibration sfc i dont think we can use this for non differentialble


    zo_eps: float = 1e-3 # eps in MeZO
    num_prefix: int = 0 # number of prefix tokens to use for MeZO
    

    # lora
    lora: bool = False # whether to use lora
    lora_alpha: int = 16
    lora_r: int = 8

    #generation
    sampling: bool = False # whether to use samplign
    temperature: float = 1.0 # temperature for sampling
    num_beams: int = 1 # number of beams for beam search
    top_k: int = None # top k for top k sampling
    top_p: float = 0.95 # top-p for generation
    max_new_tokens: int = 50 # max number of new tokens to generate
    eos_token: str = "\n" # end of stentence token

    # saving
    save_model: bool = False #whether to save the model
    no_eval: bool = False # whether to evaluate on the validation set
    tag: str = "" # tag for saving the model
    # Linear probing
    linear_probe: bool = False # whether to do linear probing
    lp_early_stopping: bool = False # whether to do early stopping for linear probing
    #untie emb/lm_head weights
    untie_emb: bool = False

    # display
    verbos:bool = False
    #non-differentiable
    non_diff: bool = False

    # auto saving when interrupted
    save_on_interrupt: bool = False
    output_dir: str = "output"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task 
        self.model, self.tokenizer = self.load_model()


    def load_model(self):

        with count_time("Load model wtih wtih HF"):
            free_in_gb = int(torch.cuda.mem_get_info()[0]/1024**3)
            config = AutoConfig.from_pretrained(self.args.model_name)
            logger.info(f"Model config: {config}")
            if self.args.untie_emb:
                logger.warm("Untying embedding and lm_head weights")
                config.tie_word_embeddings = False
            elif self.args.no_auto_device:
                # no auto device for FSDP
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config
                )

            else:
                    # Auto device loading
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config)
            model.eval()
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, )
        logger.info(f"Vocab size: {len(tokenizer)}")
        # add padding token for llamav2
         # add padding token for llamav2 
        if 'llama' in self.args.model_name:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # rewsize tokenizer embeddings length
            model.resize_token_embeddings(len(tokenizer))
        # lora
        if self.args.lora:
            from lora import LoRA
            LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha, float16=self.args.load_float16)

        return model, tokenizer
    

    # normal forward pass < need to make it so we have feedback model plus normal model w/ hotswapping weights>

    def forward(self, input_ids, option_len=None, generation=True):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """

        input_ids = torch.tensor([input_ids]).to(self.args.device)  

        # generation or classes
        if generation:
            args = self.args
            # Autogregressive Generation

            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, )
            )
            # for generation directly return the generated text
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.mode.eval()
                # get probablity distribution
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0,1:]
            logits = logits[0, :-1]
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()

            return selected_log_probs[-option_len:]
        
    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        '''
        Return the prediction on eval sample 
        '''
        pass
    #

    def train(self, train_samples, eval_samples):
        """
        Training function
        """
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]


        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer, 
                    max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True, 
                    max_new_tokens=self.args.max_new_tokens
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    #TODO un random this for actual usage :skull:
                    #correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                    correct_candidate_id = random.randint(0, 1)
                    

                
                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the 
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))
        
        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification
        # FROM TRAINER .PY
        trainer = OurTrainer(
            model=self.model, 
            args=self.args,
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
        )
        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        trainer.train(resume_from_checkpoint=last_checkpoint) 

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()
        
        # FSDP compatibility
        self.model = trainer.model 
        
        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward

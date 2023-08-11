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
    task_name: str = ""
    # number of examples


    #model loading -> need ot offlad this sumewhere

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
            if self.args.unti_emb:
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
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)
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

    def train_single_sample(

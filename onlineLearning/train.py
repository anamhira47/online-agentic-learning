from trainer import OurTrainer
import torch
import numpy as np
import random
import argparse
import tasks
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, HfArgumentParser, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification,AutoModelForMultipleChoice,OPTForSequenceClassification
from trainer import OurTrainer
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import Dataset
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from typing import Optional, Union, List, Dict, Any
from transformers.utils import PaddingStrategy
import wandb
import logging
logging.basicConfig(level=logging.INFO)
import logging
import torch
from torch import device as torch_device
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@dataclass

class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = {k: v.to(device) for k, v in batch.items()}

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
@dataclass
class DataCollatorWithPaddingAndNesting:
    """
    Collator for training
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [ff for f in features for ff in f]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch
@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "Mind2Web" # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP
    mode: str = "train" # train, eval, test, or interactive
    # Number of examples
    #num_train: int = 1000 # ICL mode: number of demonstrations; training mode: number of training samples
    #num_dev: int = 1000 # (only enabled with training) number of development samples
    #num_eval: int = 100 # number of evaluation samples
    # num_train_sets: int = 1 # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    #train_set_seed: int = 42 # designated seed to sample training samples/demos
    result_file: str = None # file name for saving performance; if None, then use the task name, model name, and config
    # num epochos
    num_train_epochs: int = 30 # number of training epochs
    # Model loading
    per_device_train_batch_size: int = 100 # batch size per device for training
    # model_name: str = "facebook/opt-125m" # HuggingFace model name
    load_float16: bool = True # load model parameters as float16
    load_bfloat16: bool = False # load model parameters as bfloat16
    load_int8: bool = False # load model parameters as int8
    max_length: int = 2048 # max length the model can take
    no_auto_device: bool = False # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    #sfc: bool = False # whether to use SFC calibration
    #icl_sfc: bool = False # whether to use SFC calibration for ICL samples
    #tag: str = "online" # tag for saving the calibration file
    # Training
    trainer: str = "zo" 
    # device: torch_device = torch_device('cuda' if torch.cuda.is_available() else 'cpu')
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo: zeroth-order (MeZO) training
    only_train_option: bool = True # whether to only train the option part of the input
    # only for decoder models
    train_as_classification: bool = False # take the log likelihood of all options and train as classification 

    # MeZO random perturbation epsilon
    zo_eps: float = 1e-3 # eps in MeZO

    # Prefix tuning
    prefix_tuning: bool = False # whether to use prefix tuning
    num_prefix: int = 5 # number of prefixes to use
    no_reparam: bool = True # do not use reparameterization trick
    prefix_init_by_real_act: bool = True # initialize prefix by real activations of random words

    # LoRA
    lora: bool = False # whether to use LoRA
    lora_alpha: int = 16 # alpha in LoRA
    lora_r: int = 8 # r in LoRA

    # Generation
    sampling: bool = False # whether to use sampling
    temperature: float = 1.0 # temperature for generation
    num_beams: int = 1 # number of beams for generation
    top_k: int = None # top-k for generation
    top_p: float = 0.95 # top-p for generation
    max_new_tokens: int = 50 # max number of new tokens to generate
    eos_token: str = "\n" # end of sentence token

    # Saving
    save_model: bool = True # whether to save the model
    no_eval: bool = False # whether to skip evaluation
    tag: str = "" # saving tag

    # Linear probing
    linear_probing: bool = False # whether to do linear probing
    lp_early_stopping: bool = False # whether to do early stopping in linear probing
    head_tuning: bool = False # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False # untie the embeddings and LM head

    # Display
    verbose: bool = True # verbose output

    # Non-diff objective
    non_diff: bool = False # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False # save model when interrupted (useful for long training)
    output_dir: str = "result" # output directory
    # extra
    learning_rate: float = 1e-7
    lr_scheduler_type: str = "constant"
    
    save_total_limit: int = 1
    train_as_classification: bool = True
    # steps
    #max_steps: int = 100
    #num_epochs: int = 1
    eval_steps: int = 5
    eval_strategy: str = 'epoch'
    # Loggging
    logging_dir: str = "logs"
    logging_steps: int = 10
    report_to: str = "wandb"
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.1
model_name = 'bert-large-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"acc": (preds == label_ids).astype(np.float32).mean().item()}
def preprocess_function(examples):
    ending_names = ["ending0", "ending1", "ending2", "ending3"]
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["sent2"]
    second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]
    
    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    
    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # Un-flatten
    return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

def main():
    global tokenizer
    if 'roberta' in model_name:
        model = AutoModelForMultipleChoice.from_pretrained(model_name)
        if torch.cuda.is_available():
            model = model.to('cuda')
    elif 'bert' in model_name:
        model = AutoModelForMultipleChoice.from_pretrained(model_name)
        if torch.cuda.is_available():
            model = model.to('cuda')
    elif 'opt' in model_name:
        # Need to implement OPt multiple choice model
        pass
    elif 'llama' in model_name:
        # Need to implement llama multiple choice model
        pass

    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    swag = load_dataset("swag", "regular")
    swag.set_format("torch", device="cuda")
    swag['train'] = swag['train'].select(range(1000))
    swag['validation'] = swag['validation'].select(range(1000))
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # replace with your model's tokenizer
   
        # Return the processed data as a dictionary
    tokenized_swag = swag.map(preprocess_function, batched=True)
    # Print the first two examples from tokenized_swag
    #print(tokenized_swag['train'][:2])
# Apply the preprocessing function to the dataset
    #dataset = dataset.map(preprocess_function, batched=True)
    # Print some examples from swag
    

    
    
    # Split the dataset into train and eval datasets
    #print(tokenized_swag['train'])
    train_dataset, eval_dataset = tokenized_swag['train'], tokenized_swag['validation']
    trainer = OurTrainer(
            model=model, 
            args=args,
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer),
            compute_metrics=compute_metrics,

        )
    
    trainer.train()

    # Evaluate the model


    if args.save_model:
        trainer.save_model()
    
if __name__ == '__main__':

    main()
from templates import *
from utils import temp_seed
import json
import os
from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Union
import string
import random
import datasets
import sys
import numpy as np
import logging
from sklearn.model_selection import train_test_split

# from mind2webloader import get_data_split, TextMultiChoiceDataset
import pickle
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_task(task_name):
    aa = task_name.split("__")
    if len(aa) == 2:
        task_group, subtask = aa
    else:
        task_group = aa[0]
        subtask = None
    class_ = getattr(sys.modules[__name__], f"{task_group}Dataset")
    instance = class_(subtask)
    return instance


@dataclass
class Sample:
    id: int = None
    data: dict = None
    correct_candidate: Union[str, List[str]] = None
    candidates: List[str] = None


class Dataset:
    mixed_set = False
    train_sep = "\n\n"
    generation = False # whether this is a generation task

    def __init__(self, subtask=None, **kwargs) -> None:
        self.subtask = subtask
    
    def get_task_name(self):
        return self.subtask
        
    def load_dataset():
        raise NotImplementedError
    
    def get_template(self, template_version=0):
       templates = {0: Template}
       return templates[template_version]
   
    def build_sample(self, example):
        return 
     
    def sample_train_sets(self, num_train=32, num_dev=None, num_eval=None, num_train_sets=None, seed=None):
        if seed is not None:
            # one train/demo set using the designated seed
            seeds = [seed]
        elif num_train_sets is not None:
            # num_train_sets train/demo sets
            seeds = list(range(num_train_sets))
        else: 
            # one train/demo set per evaluation sample
            assert num_dev is None # not supported
            len_valid_samples = len(self.samples["valid"]) if num_eval is None else num_eval
            with temp_seed(0):
                seeds = np.random.randint(0, 10000, len_valid_samples)

        train_samples = [] 
        for i, set_seed in enumerate(seeds):
            if self.mixed_set:
                raise NotImplementedError
                train_samples.append(self.sample_subset(data_split="valid", seed=set_seed, num=num_train, exclude=i))
            else:
                if num_dev is not None:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train+num_dev)) # dev set is included at the end of train set
                    if num_train + num_dev > len(self.samples["train"]):
                        logger.warn("num_train + num_dev > available training examples")
                else:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train))
                if num_dev is not None:
                    logger.info(f"Sample train set {len(train_samples[-1])}/{len(self.samples['train'])}")
                    logger.info(f"... including dev set {num_dev} samples")
        return train_samples

    def sample_subset(self, data_split="train", seed=0, num=100, exclude=None):
        with temp_seed(seed):
            samples = self.samples[data_split] 
            lens = len(samples)
            index = np.random.permutation(lens).tolist()[:num if exclude is None else num+1]
            if exclude is not None and exclude in index:
                index.remove(exclude)
            else:
                index = index[:num]
            return [samples[i] for i in index]
    
    @property
    def valid_samples(self):
        return self.samples["valid"]


class SST2Dataset(Dataset):
    train_sep = "\n\n"
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        d = load_dataset('glue', 'sst2')
        train_d = d["train"]
        validation_d = d["validation"]
        
        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: SST2Template}[template_version]()
        
    
class CopaDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        train_examples = load_dataset('super_glue', "copa")["train"]
        valid_examples = load_dataset('super_glue', "copa")["validation"]
    
        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example):
        sample = \
            Sample(
                id=example["idx"],
                data=example,
                candidates=[example["choice1"], example["choice2"]],
                correct_candidate=example[f"choice{example['label'] + 1}"],
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: CopaTemplate}[template_version]()


class BoolQDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("boolq")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=["Yes", "No"],
                correct_candidate="Yes" if example["answer"] else "No",
            )
        
        return sample
    
    def get_template(self, template_version=2):
        return {0: BoolQTemplate, 1: BoolQTemplateV2, 2: BoolQTemplateV3}[template_version]()


class MultiRCDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "multirc")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: MultiRCTemplate}[template_version]()


class CBDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "cb")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1, 2],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: CBTemplate}[template_version]()


class WICDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
        
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wic")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WICTemplate}[template_version]()


class WSCDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wsc.fixed")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WSCTemplate}[template_version]()


class ReCoRDDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "record")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=example['entities'],
                correct_candidate=example['answers']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: ReCoRDTemplateGPT3}[template_version]()


class RTEDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "rte")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: RTETemplate}[template_version]()

 
class SQuADDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()
        
    def load_dataset(self):
        dataset = load_dataset("squad")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers']['text']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "title": example['title'],
                "context": example['context'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )
        
    def get_template(self, template_version=0):
        return {0: SQuADv2Template}[template_version]()
# Generic multi choice dataset for senidng from the onlinelearning

# non diff version of Mind2web
class Mind2WebNonDiffDataset(Dataset):
    metric_name = "f1"
    generation = True
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()
    def load_dataset(self):
        print('Getting data')
        data_path = '/home/anam.hira/Mind2Web'
        train_split_file = 'data/train/*.json'
        test_split_files = {
        'test_task': 'data/test_task/*.json',
        'test_website': 'data/test_website/*.json',
        'test_domain': 'data/test_domain/*.json'
        }
        score_file = '/home/anam.hira/Mind2Web/scores_all_data.pkl'
        with open(score_file, "rb") as f:
            candidate_results = pickle.load(f)
        test_dataset_dict = {}

        i = 0
        for test_key, test_split_file in test_split_files.items():
            if i >= 1:
                break
            i += 1
            test_data = get_data_split(
                data_path,
                test_split_file,
                candidate_results=candidate_results,
            )
            print("Got test data")
            test_dataset_dict[test_key] = TextMultiChoiceDataset(
                test_data,
                neg_ratio=0.5,  # Specify the desired neg_ratio
                num_candidates=5,  # Specify the desired num_candidates
                max_context_len=512,  # Specify the desired max_context_len
                mode='multichoice',  # Specify the desired mode
            )
        #dataset = load_dataset("mind2web")
        #all_samples = dataset["train"]
        
        for key, dataset in test_dataset_dict.items():
            train_samples, valid_samples = train_test_split(dataset, test_size=0.1)
            self.samples = {"train": [self.build_sample(example,idx) for idx, example in enumerate(train_samples)], 
                                 "valid": [self.build_sample(example, idx) for idx, example in enumerate(valid_samples)]}
            
    def build_sample(self, example, idx):
        context = example['context']
        question = example['input']
        answer = example['output']
        return Sample(
            id=idx,
            data={"context": context,
                    "question": question,
                    "answer": answer},
            candidates=None,
            correct_candidate=answer
        )
    def get_template(self, template_version=0):
        return {0: Mind2WebNonDiffTemplate}[template_version]()

        
    


class Mind2WebDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        pass
        
    def load_dataset(self):
        print('Getting data (load_datset)')
        data_path = '/home/anam.hira/Mind2Web'
        train_split_file = 'data/train/*.json'
        test_split_files = {
        'test_task': 'data/test_task/*.json',
        'test_website': 'data/test_website/*.json',
        'test_domain': 'data/test_domain/*.json'
        }
        score_file = '/home/anam.hira/Mind2Web/scores_all_data.pkl'
        with open(score_file, "rb") as f:
            candidate_results = pickle.load(f)
        test_dataset_dict = {}

        i = 0
        for test_key, test_split_file in test_split_files.items():
            if i >= 1:
                break
            i += 1
            test_data = get_data_split(
                data_path,
                test_split_file,
                candidate_results=candidate_results,
            )
            print("Got test data (load_dataset)")
            test_dataset_dict[test_key] = TextMultiChoiceDataset(
                test_data,
                neg_ratio=0.5,  # Specify the desired neg_ratio
                num_candidates=5,  # Specify the desired num_candidates
                max_context_len=512,  # Specify the desired max_context_len
                mode='multichoice',  # Specify the desired mode
            )
        #dataset = load_dataset("mind2web")
        #all_samples = dataset["train"]
        print(f'Building samples (load_dataset) Number of sets:{len(test_dataset_dict)}')
        for key, dataset in test_dataset_dict.items():
            train_samples, valid_samples = train_test_split(dataset, test_size=0.1)
            self.samples = {"train": [self.build_sample(example,idx) for idx, example in enumerate(train_samples)], 
                                 "valid": [self.build_sample(example, idx) for idx, example in enumerate(valid_samples)]}

    def build_sample(self, example, idx):
        context = example['context']
        question = example['input']
        answer = example['output']
        #candidates = 
        letters = ['B','C','D']
        operation_types = ['CLICK', 'TYPE', 'SELECT']
        candidates = [f"{letter}.\nAction: {operation}" for letter in letters for operation in operation_types]
        # add none of the above option
        candidates.append(f"A. None")
        # remove the text after 'Value:' for the answer if it has a value
        answer = answer.split('Value:')[0].strip() if 'Value:' in answer else answer


        return Sample(
            id=idx,
            data={"context": context,
                  "question": question,
                    "answer": answer},
            candidates=candidates,
            correct_candidate=answer
        )
    


    def get_template(self, template_version=0):
        return {0: Mind2WebTemplate}[template_version]()
    




class GenericMultiChoiceDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()
        pass
    def load_dataset(self):
        '''
        Implementation with
        '''
        pass
    def build_sample(self, example):
        answer = example['correct_candidate']
        assert len(answer) > 0
        return Sample(
            id=example["id"],
            data=example['data'],
            candidates=example['candidates'],
            correct_candidate=answer
        )
    
    def get_template(self, template_version=0):
        return {0: GenericMultiChoiceTemplate}[template_version]()


    


class DROPDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()
        
    def load_dataset(self):
        dataset = load_dataset("drop")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers_spans']['spans']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "context": example['passage'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )
        
    def get_template(self, template_version=0):
        return {0: DROPTemplate}[template_version]()

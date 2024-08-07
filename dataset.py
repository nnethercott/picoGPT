import functools
import random
import re
import os

import datasets
import torch
from datasets import interleave_datasets, load_dataset
from torch.utils.data import DataLoader, Dataset


def remove_lines_with_substrings(text, substrings):
    # Create a regex pattern that matches lines containing any of the substrings
    pattern = re.compile(r'^(.*({}).*)$'.format('|'.join(map(re.escape, substrings))), re.MULTILINE)
    # Substitute the matching lines with an empty string
    cleaned_text = re.sub(pattern, '', text)
    # Remove any leading or trailing whitespace from the result
    return cleaned_text.strip()

def collate_fn(inputs):
    return torch.tensor(inputs)

# right now assumes bsz=1
def sft_collate_fn(inputs):
  # used in batch_size = 1 training 
  inputs = inputs[0]
  return {
    'prompt_len': torch.tensor(inputs['prompt_len']),
    'input_ids': torch.tensor(inputs['input_ids']).unsqueeze(0) # bsz 1 
    }

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_starcoder(lang, tok, rank = 0):
    # here `text` key in return dict is prompt+answer
    # NOTE: do tok(prompt+answer) and len(tok(prompt))

    data = load_dataset("bigcode/starcoderdata", data_dir=lang, split="train", streaming=True, token=os.environ["HF_ACCESS_TOKEN"])
    BLOCK_SIZE = 200000
    data = data.skip(1500000+BLOCK_SIZE*rank).take(BLOCK_SIZE)

    def dataset_generator(dataset):
      yield from dataset

    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))
    data = data.map(
      lambda x: {**x, "input_ids": [tok.encode(remove_lines_with_substrings(y, ["<gh_stars>", "<filename>"]))+[tok.eos_token_id] for y in x["content"]]},
      batched=True,
    )
    data = data.filter(
      lambda x: [len(y) <= 384 and len(y)>=16 for y in x["input_ids"]], batched=True
    )

    data = [{'prompt_len': 0, 'input_ids':i} for i in data['input_ids']]
    ds = CustomDataset(data)

    return ds


def load_evol_py(tok, rank = 0, world_size = 1):
    data = load_dataset("mlabonne/Evol-Instruct-Python-26k", split="train")

    #filter to samples containing code
    data = data.filter(lambda x: len(re.findall(r'```(.*?)```', x['output'], re.DOTALL))>0)

    def apply_convo(examples):
        #filter out explanations, only keep the code 
        outputs = [re.findall(r'```(.*?)```', o, re.DOTALL)[0] for o in examples['output']]
        #outputs = ["ASSISTANT: " + re.sub(r'python','', o) for o in outputs]
        outputs = [re.sub(r'python','', o) for o in outputs]

        # no comments pls
        outputs = [re.sub(r'^\s*#.*\n?', '', o, flags=re.MULTILINE) for o in outputs]

        #prompts = ["USER: "+ i +'\n' for i in examples['instruction']]
        prompts = [i +'\n\n' for i in examples['instruction']]

        prompt_lens = [len(tok.encode(p)) for p in prompts]
        input_ids = [tok.encode(f'{p}{o}')+ [tok.eos_token_id] for p, o in zip(prompts, outputs)]
        
        # updata examples keys 
        examples['prompt_len'] = prompt_lens 
        examples['input_ids'] = input_ids
        return examples 

    data = data.map(
        apply_convo, 
        batched=True,
    )

    data = data.filter(
      lambda x: [len(y) <= 384 and len(y)>16 for y in x["input_ids"]], batched=True
    )

    data = [{'prompt_len': x['prompt_len'], 'input_ids': x['input_ids']} for x in data]
    bsz = len(data)//world_size 
    data = data[bsz*rank : bsz*(rank+1)]

    ds = CustomDataset(data) 
    return ds


def load_ultrachat(tok, rank = 0, world_size = 1):
    data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    def apply_convo(examples):
        # each sample has potentially many conversations, let's only keep the first 
        msgs = [msgs[:2] for msgs in examples['messages']]
        outputs = ["ASSISTANT: " + msg[1]['content'] for msg in msgs]
        prompts = ["USER: "+ msg[0]['content'] +'\n' for msg in msgs]

        prompt_lens = [len(tok.encode(p)) for p in prompts]
        input_ids = [tok.encode(f'{p}{o}')+ [tok.eos_token_id] for p, o in zip(prompts, outputs)]
        
        # updata examples keys 
        examples['prompt_len'] = prompt_lens 
        examples['input_ids'] = input_ids
        return examples 

    data = data.map(
        apply_convo, 
        batched=True,
    )

    data = data.filter(
      lambda x: [len(y) <= 384 and len(y)>16 for y in x["input_ids"]], batched=True
    )

    data = [{'prompt_len': x['prompt_len'], 'input_ids': x['input_ids']} for x in data]
    bsz = len(data)//world_size 
    data = data[bsz*rank : bsz*(rank+1)]

    ds = CustomDataset(data) 
    return ds


def load_alpaca_instruct(tok, rank = 0, world_size=1):
    data = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

    # some entries contain 'input': Not available -> just filter to empty 
    def apply_convo(examples):
        inputs = ['' if i.startswith('Not') else i for i in examples['input']]
        outputs = examples['output']
        instruction = examples['instruction']

        prompts = [f'{i}\nInput: {p}\n' if len(p)!=0 else f'{i}\n' for i,p in zip(instruction, inputs)]
       # prompts = ["USER: " + p + '\n' for p in prompts]
       # outputs = ["ASSISTANT: " + o for o in outputs]
        prompts = [p + '\n\n' for p in prompts]
        outputs = [o for o in outputs]

        # no <EOS> token added
        prompt_lens = [len(tok.encode(p)) for p in prompts]
        
        input_ids = [tok.encode(f'{p}{o}')+ [tok.eos_token_id] for p, o in zip(prompts, outputs)]
        
        # updata examples keys 
        examples['prompt_len'] = prompt_lens 
        examples['input_ids'] = input_ids
        return examples 

    data = data.map(
        apply_convo, 
        batched=True,
    )

    data = data.filter(
      lambda x: [len(y) <= 384 and len(y)>16 for y in x["input_ids"]], batched=True
    )

    data = [{'prompt_len': x['prompt_len'], 'input_ids': x['input_ids']} for x in data]
    bsz = len(data)//world_size 
    data = data[bsz*rank : bsz*(rank+1)]

    ds = CustomDataset(data) 
    return ds

#tiny shakespeare 
#text = load_dataset("karpathy/tiny_shakespeare")['train'][0]['text']
#tokens = tok.encode(text)
#tokens = tokens[:-(len(tokens)%model_config.block_size)]
#tokens = [tokens[i*model_config.block_size:(i+1)*model_config.block_size] for i in range(len(tokens)//model_config.block_size)]
#tokens = tokens[:500]
#ds = CustomDataset(tokens)
#dl = DataLoader(
#    ds, batch_size=train_config.batch_size, shuffle=False, collate_fn=collate_fn
#)


def load_slimpajama(tok, rank=0):
  data = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
  BLOCK_SIZE=200000
  texts = data.skip(8000000+rank*BLOCK_SIZE).take(BLOCK_SIZE)

  # preprocessing
  # https://stackoverflow.com/questions/76227219/can-i-convert-an-iterabledataset-to-dataset
  def dataset_generator(dataset):
      yield from dataset

  texts = datasets.Dataset.from_generator(functools.partial(dataset_generator, texts))
  texts = texts.map(
      lambda x: {**x, "tokens": [tok.encode(y)+[tok.eos_token_id] for y in x["text"]]},
      batched=True,
  )
  texts = texts.filter(
      lambda x: [len(y) >= 16 and len(y)<=384 for y in x["tokens"]], batched=True
  )
  tokens = texts["tokens"]
  data = [{'prompt_len': 0, 'input_ids': t} for t in tokens]

  # dataset object
  ds = CustomDataset(data)

  return ds


class InterpolatedDataset:
  def __init__(self, *datasets):
    """
    each dataset instance should be a dict with keys `data`, `target_ratio`, and `is_main`
    each `data` is a torch.utils.data.Dataset instance, `is_main` indicates if dataset is backbone  
    """
    datasets = [*datasets]

    # allow option to not interpolate 
    #assert sum([item['is_main'] for item in datasets]) == 1

    self.datasets = datasets

  @property 
  def sampling_ratios(self):
    ratios = [ds['target_ratio'] for ds in self.datasets]
    return [r/sum(ratios) for r in ratios]

  def generate(self, saturation_steps: int)->torch.utils.data.Dataset:
    """
    exhausts provided datasets and linearly increases sampling ratios until `saturation_steps` reached 
    """
    ds_steps = [0 for _ in self.datasets]
    merged_data = [] 

    # initalize sampling ratios as ~onehot(is_main)
    is_main = [ds['is_main'] for ds in self.datasets]
    ratios = [r/saturation_steps if not i else r for i,r in zip(is_main, self.sampling_ratios)]

    steps = 0 
    while self.datasets:
      idx = random.choices(range(len(self.datasets)), weights = ratios, k=1)[0]

      ds = self.datasets[idx]['data']
      step = ds_steps[idx]
      merged_data.append(ds[step])

      # increment dataset-local index 
      ds_steps[idx]+=1

      # if exhausted remove
      if ds_steps[idx] >= len(ds):
        del ratios[idx]
        del self.datasets[idx]
        del ds_steps[idx]
      
      # update ratios
      if steps<saturation_steps:
        ratios = [r*(steps+1)/saturation_steps if not i else r for r,i in zip(self.sampling_ratios, is_main)]
      steps+=1

    return CustomDataset(merged_data)


from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# starcoder = load_starcoder("python", tok)
# slimpajama = load_slimpajama(tok)
# alpaca = load_alpaca_instruct(tok)
# chat = load_ultrachat(tok)
evol = load_evol_py(tok)

# ds = InterpolatedDataset({'data': starcoder, 'target_ratio':1, 'is_main': False}, {'data': slimpajama, 'target_ratio': 2, 'is_main': True})


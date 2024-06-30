from torch.utils.data import DataLoader, Dataset
import datasets
from datasets import load_dataset 
import torch 
import functools

def collate_fn(inputs):
    return torch.tensor(inputs)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_training_data(model_config, train_config, tok, rank=0):
  data = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
  BLOCK_SIZE=500000
  texts = data.skip(2000000+rank*BLOCK_SIZE).take(BLOCK_SIZE)

  # preprocessing
  # https://stackoverflow.com/questions/76227219/can-i-convert-an-iterabledataset-to-dataset
  def dataset_generator(dataset):
      yield from dataset

  texts = datasets.Dataset.from_generator(functools.partial(dataset_generator, texts))
  texts = texts.map(
      lambda x: {**x, "tokens": [tok.encode(y)[: model_config.block_size] for y in x["text"]]},
      batched=True,
  )
  texts = texts.filter(
      lambda x: [len(y) == model_config.block_size for y in x["tokens"]], batched=True
  )
  tokens = texts["tokens"]


  # dataset object
  ds = CustomDataset(tokens)
  dl = DataLoader(
      ds, batch_size=train_config.batch_size, shuffle=False, collate_fn=collate_fn
  )

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
  return dl 

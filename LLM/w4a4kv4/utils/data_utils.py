# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import random
from typing import Any, Dict

import datasets
import torch
import transformers
from datasets import load_from_disk


def get_wikitext2(nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, eval_mode=False):
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    if eval_mode:
        testdata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "test"
        ]
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    else:
        traindata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "train"
        ]
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4(nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, eval_mode=False):
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    alldata = load_from_disk('datasets/allenai/c4/allenai--c4')
    traindata = alldata['train']
    valdata = alldata['validation']

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # import random
    # random.seed(0)
    # valenc = []
    # for _ in range(256):
    #     while True:
    #         i = random.randint(0, len(valdata) - 1)
    #         tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
    #         if tmp.input_ids.shape[1] >= seqlen:
    #             break
    #     i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     valenc.append(tmp.input_ids[:, i:j])
    # valenc = torch.hstack(valenc)

    # class TokenizerWrapper:

    #     def __init__(self, input_ids):
    #         self.input_ids = input_ids

    # valenc = TokenizerWrapper(valenc)

    # if eval_mode:
    #     return valenc
    return trainloader

save_seed = False

def get_c4_val(tokenizer, seqlen=2048):
    valdata = load_from_disk('OurGPTQ/GPTAQ/eval_my/ppl_datasets/allenai/c4/allenai--c4/validation')
    import random
    random.seed(0)
    # index_list = random.sample(list(range(len(val_data))), 256)
    valenc = []
    if save_seed:
        index_list = []
        tmp_list = []
    else:
        index_list = torch.load('OurGPTQ/GPTAQ/eval_my/c4_seed_index.pth')
        tmp_list = torch.load('OurGPTQ/GPTAQ/eval_my/c4_seed_tmp.pth')
    for _ in range(256):
        if save_seed:
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
                if tmp.input_ids.shape[1] > seqlen:
                    tmp_list.append(i)
                    break
            # tmp = tokenizer(val_data[index_list[_]]['text'], return_tensors='pt')
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            index_list.append(i)
        else:
            i = index_list[_]
            tmp = tmp_list[_]
        j = i + seqlen
        if save_seed:
            valenc.append(tmp.input_ids[:, i:j])
        else:
            valenc.append(tokenizer(valdata[tmp]['text'], return_tensors='pt').input_ids[:, i:j])
    if save_seed:
        torch.save(index_list, '../eval_my/c4_seed_index.pth')
        torch.save(tmp_list, '../eval_my/c4_seed_tmp.pth')
        breakpoint()
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return valenc 

class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size: int = 1024) -> None:
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d))

        grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, Any]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

import numpy as np
import torch
from datasets import load_dataset, load_from_disk


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    traindata = load_from_disk('eval_my/ppl_datasets/wikitext/wikitext-2-raw-v1/train')
    # testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testdata = load_from_disk('eval_my/ppl_datasets/wikitext/wikitext-2-raw-v1/test')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model, save_seed=False):
    from datasets import load_dataset
    import torch
    # traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', use_auth_token=False)
    # valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', use_auth_token=False)
    alldata = load_from_disk('eval_my/ppl_datasets/allenai/c4/allenai--c4')
    traindata = alldata['train']
    valdata = alldata['validation']

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    import random
    random.seed(seed)
    trainloader = []
    
    model_name = model.lower()
    if 'llama-3.2' in model_name:
        model_type = 'llama3.2_1B'
    elif 'llama-3' in model_name:
        model_type = 'llama3_8B'
    elif 'llama-2' in model_name:
        model_type = 'llama-2_13B'
   
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    train_doc_index_path = f'GPTQ/utils/{model_type}_c4_train_doc_index.pth'
    train_pos_index_path = f'GPTQ/utils/{model_type}_c4_train_pos_index.pth'
    
    if save_seed:
        train_indices = []
        train_positions = []
        
    for _ in range(nsamples):
        if save_seed:
            
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    train_indices.append(i)
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            train_positions.append(i)
        else:
            
            train_indices = torch.load(train_doc_index_path)
            train_positions = torch.load(train_pos_index_path)
            i_sample = train_indices[_]
            trainenc = tokenizer(traindata[i_sample]['text'], return_tensors='pt')
            i = train_positions[_]
                
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    if save_seed:
        torch.save(train_indices, train_doc_index_path)
        torch.save(train_positions, train_pos_index_path)

    random.seed(0)
    valenc = []
    
    doc_index_path = f'GPTQ/utils/{model_type}_c4_val_doc_index.pth'  
    pos_index_path = f'GPTQ/utils/{model_type}_c4_val_pos_index.pth'  
    
    if save_seed:
        doc_indices = []  
        pos_indices = [] 
    else:
        doc_indices = torch.load(doc_index_path)
        pos_indices = torch.load(pos_index_path)
    
    for _ in range(256):
        if save_seed:
            
            while True:
                i = random.randint(0, len(valdata) - 1) 
                tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
                if tmp.input_ids.shape[1] > seqlen:
                    doc_indices.append(i) 
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1) 
            pos_indices.append(i) 
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j]) 
        else:
            doc_id = doc_indices[_]  
            pos = pos_indices[_]    
            j = pos + seqlen
            
            tmp = tokenizer(valdata[doc_id]['text'], return_tensors='pt')
            valenc.append(tmp.input_ids[:, pos:j])
    
    if save_seed:
        torch.save(doc_indices, doc_index_path)
        torch.save(pos_indices, pos_index_path)
        
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

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

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)

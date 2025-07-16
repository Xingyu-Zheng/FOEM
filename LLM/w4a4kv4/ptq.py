# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger
import os 
import json
import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq

log: Logger = utils.get_logger("spinquant")
def generate_text(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
    formatted_prompt = f"""Below is a conversation between a helpful AI assistant and a user.

User: {prompt}

Assistant:"""
    
    model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()
def process_questions(model, tokenizer):
    questions_file = os.path.abspath('OurGPTQ/questions.jsonl')
    output_file = os.path.abspath('OurGPTQ/gptq.txt')
    
    with open(questions_file, 'r') as f:
        questions = [json.loads(line) for line in f]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(f"Question ID: {question['question_id']}\n")
            f.write(f"Category: {question['category']}\n")
            f.write(f"Question: {question['text']}\n")
            f.write("Answer:\n")
            
            log.info(f"Processing question {question['question_id']}...")
            answer = generate_text(model, tokenizer, question['text'])
            
            f.write(answer)
            f.write("\n" + "="*50 + "\n\n")


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=18))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    # model.cuda()
    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = training_args.model_max_length
    model.cuda()
    if local_rank == 0:
        # log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False
    if local_rank == 0:
        log.info("Starting to process questions...")
        process_questions(model, tokenizer)
        log.info("Question processing completed!")

    dist.barrier()


if __name__ == "__main__":
    train()

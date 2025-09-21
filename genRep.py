import os
import json
import torch
from typing import Any, Union,Optional, List
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, PretrainedConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
# import bitsandbytes as bnb
# import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from torch import device, Tensor, nn
import numpy as np
from tqdm.autonotebook import tqdm, trange
from accelerate import Accelerator
from torch.amp import autocast
from peft.tuners.lora.layer import LoraLayer
import sys
import logging
from torch.nn import CrossEntropyLoss
# from models import MistralBiModel, LlamaBiModel, MistralBiForCausalLM, LlamaBiForCausalLM


# class StreamToLogger:
#     def __init__(self, logger, level):
#         self.logger = logger
#         self.level = level
#         self.buffer = ""

#     def write(self, message):
#         # 避免重复打印空行
#         if message != "\n":
#             self.logger.log(self.level, message.strip())

#     def flush(self):
#         pass  # Required for compatibility
        
# class Logger(logging.Logger):
#     def __init__(self):
#         super().__init__("target")

#     def add_stream_handler(self):
#         sh = logging.StreamHandler()
#         sh.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
#         self.addHandler(sh)

#     def add_file_handler(self, save_dir, rank):
#         # rank = dist.get_rank()
#         fh = logging.FileHandler(save_dir + f"/log_rank_{rank}.txt", "a")
#         fh.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
#         self.addHandler(fh)

#     def set_verbosity_info(self):
#         self.setLevel(logging.INFO)

#     def set_verbosity_error(self):
#         self.setLevel(logging.ERROR)

class GenRep(nn.Module):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                 accelerator = None, logger=None, output_dir=None,
                 pooling_mode: str = "mean",
                 prompt: str = '*sent_0*', max_length: int = 256,
                 simcse:bool = False, simcse_dropout: float = 0.3):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.max_length = max_length
        self.text_max_length = max_length - 10   #prompt length is  10
        self.prompt = prompt
        self.pooling_mode = pooling_mode
        self.simcse = simcse
        self.simcse_dropout = simcse_dropout
        self.accelerator = accelerator or Accelerator()
        self.logger=logger
        self.output_dir=output_dir


    @classmethod
    def _get_model_class(cls, config_class_name, enable_bidirectional):
        if not enable_bidirectional:
            return AutoModelForCausalLM
        else:
            from models import bidirectional_get_model_class
            return bidirectional_get_model_class(config_class_name)
        # if config_class_name == "MistralConfig":
        #     return MistralBiForCausalLM
        # elif config_class_name == "LlamaConfig":
        #     return LlamaBiForCausalLM
        # elif config_class_name == "GemmaConfig":
        #     return GemmaBiModel
        # elif config_class_name == "Qwen2Config":
        #     return Qwen2BiModel
        # else:
        #     raise ValueError(
        #         f"{config_class_name} is not supported yet with bidirectional models."
        #     )
            
    @classmethod
    def from_pretrained(
        cls,
        base_model_path,
        logger=None,
        lora_config: dict = {},
        checkpoint_path: str = '',
        pre_checkpoint_path: str = '',
        pre_checkpoint_path1: str = '',
        pre_checkpoint_path2: str = '',
        trainable: bool = False,
        enable_bidirectional: bool = False,
        follow_mntp: bool = False,
        follow_llara: bool = False,
        follow_llara_double: bool = False,
        simcse: bool = False,
        simcse_dropout: float = 0.3,
        **kwargs,
    ):

        keys = ["pooling_mode", "max_length", "prompt", "output_dir", "simcse_dropout", "simcse"]
        encoder_args = {
            key: kwargs.pop(key, None) for key in keys if kwargs.get(key) is not None
        }

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(base_model_path)
        config_class_name = config.__class__.__name__
        
        model_class = cls._get_model_class(
            config_class_name, enable_bidirectional=enable_bidirectional
        )
        
        if trainable:
            config.gradient_checkpointing = True
            base_model = model_class.from_pretrained(
                base_model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager"
            )
            
            if follow_mntp:
                peft_target = base_model.get_model_for_peft()
                peft_target = PeftModel.from_pretrained(
                    peft_target,
                    pre_checkpoint_path,
                )
                peft_target = peft_target.merge_and_unload()
                if hasattr(peft_target, 'peft_config'):
                    del peft_target.peft_config
                base_model.set_model_for_peft(peft_target)

            if follow_llara:
                peft_target = PeftModel.from_pretrained(
                    base_model,
                    pre_checkpoint_path,
                )
                logger.info("merge model!!")
                base_model = peft_target.merge_and_unload()

            if follow_llara_double:
                peft_target = PeftModel.from_pretrained(
                    base_model,
                    pre_checkpoint_path,
                )
                logger.info("merge model!!")
                base_model = peft_target.merge_and_unload()

                peft_target = PeftModel.from_pretrained(
                    base_model,
                    pre_checkpoint_path1,
                )
                logger.info("merge model!!")
                base_model = peft_target.merge_and_unload()

                # peft_target = PeftModel.from_pretrained(
                #     base_model,
                #     pre_checkpoint_path2,
                # )
                # logger.info("merge model!!")
                # base_model = peft_target.merge_and_unload()
                
            if simcse:
                lora_config = LoraConfig(
                    r=lora_config['r'],                             
                    lora_alpha=lora_config['lora_alpha'],           
                    target_modules=["q_proj", "v_proj", "k_proj"],
                    # target_modules=["q_proj", "v_proj", "k_proj","o_proj","gate_proj","up_proj","down_proj"],
                    lora_dropout=lora_config['lora_dropout'],       
                    bias=lora_config['bias'], 
                    task_type=None
                )
            else:
                lora_config = LoraConfig(
                    r=lora_config['r'],                             
                    lora_alpha=lora_config['lora_alpha'],           
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    # target_modules=["q_proj", "v_proj", "k_proj","o_proj","gate_proj","up_proj","down_proj"],
                    lora_dropout=lora_config['lora_dropout'],       
                    bias=lora_config['bias'], 
                    task_type="CAUSAL_LM"
                )
            
            logger.info(lora_config)
            model = get_peft_model(base_model, lora_config)
    
            from peft.tuners.lora import LoraLayer
            model.print_trainable_parameters()

            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module.to(torch.float32)
                if 'norm' in name:
                    module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        module.to(torch.float32)
        else:
            base_model = model_class.from_pretrained(
                base_model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager"
            )
            
            if follow_mntp:
                peft_target = base_model.get_model_for_peft()
                peft_target = PeftModel.from_pretrained(
                    peft_target,
                    pre_checkpoint_path,
                )
                peft_target = peft_target.merge_and_unload()
                if hasattr(peft_target, 'peft_config'):
                    del peft_target.peft_config
                base_model.set_model_for_peft(peft_target)

                peft_target = base_model.get_model_for_peft()
                peft_target = PeftModel.from_pretrained(
                    peft_target,
                    checkpoint_path,
                )
                peft_target = peft_target.merge_and_unload()
                if hasattr(peft_target, 'peft_config'):
                    del peft_target.peft_config
                base_model.set_model_for_peft(peft_target)
                print(type(base_model))
                checkpoint_path=""
                
            if follow_llara:
                peft_target = PeftModel.from_pretrained(
                    base_model,
                    pre_checkpoint_path,
                )
                print("merge model!!")
                base_model = peft_target.merge_and_unload()


            if follow_llara_double:
                peft_target = PeftModel.from_pretrained(
                    base_model,
                    pre_checkpoint_path,
                )
                print("merge model!!")
                base_model = peft_target.merge_and_unload()

                peft_target = PeftModel.from_pretrained(
                    base_model,
                    pre_checkpoint_path1,
                )
                print("merge model!!")
                base_model = peft_target.merge_and_unload()

                # peft_target = PeftModel.from_pretrained(
                #     base_model,
                #     pre_checkpoint_path2,
                # )
                # print("merge model!!")
                # base_model = peft_target.merge_and_unload()
                
            if checkpoint_path:
                model = PeftModel.from_pretrained(
                    base_model, 
                    checkpoint_path,
                    torch_dtype=torch.bfloat16
                )
            else:
                model = base_model
            model.config.enable_input_require_grads = False 
            

            
        config = {key: value for key, value in encoder_args.items()}
        # logger.info(f"config:{config}")
        return cls(model=model, tokenizer=tokenizer, logger=logger, **config)

    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None, 
                labels: torch.Tensor = None, 
                return_attentions=False, **kwargs):
        """
        Processes input sentences, tokenizes them, and applies the model to get embeddings.
        """
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # for i, input_id in enumerate(input_ids):
            #     text = self.tokenizer.decode(input_id, skip_special_tokens=False)
                # print(f"Sample {i} text:")
                # print("input_id：", input_id)
            #     print("text：", text)
            if self.simcse:
                print("doing simcse")
                outputs = self.model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True, 
                    return_dict=True,
                    use_cache=False,
                    attention_dropout=self.simcse_dropout,
                    output_attentions=return_attentions
                )
            else:
                outputs = self.model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True, 
                    return_dict=True,
                    use_cache=False,
                    output_attentions=return_attentions
                )
            pooled_output = self._pool(input_ids, attention_mask, outputs.hidden_states[-1])
            if labels is not None:
            # Shift so that tokens < n predict n
                logits = self.model.lm_head(outputs.hidden_states[-1])
                # print(f"logits:{logits}")
                # print(f"labels:{labels}")
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                ar_loss = loss_fct(shift_logits, shift_labels)
                return pooled_output, ar_loss
            
    
            elif return_attentions:
                attention = outputs.attentions[-1]
                for i in range(1, len(outputs.attentions)):
                    attention += outputs.attentions[i].detach()
                attention /= len(outputs.attentions)
                return pooled_output, attention
            else:
                return pooled_output
            
    def forward_freeze(self, input_ids: torch.Tensor, 
                       attention_mask: torch.Tensor = None, 
                       return_attentions=False, **kwargs):
        """
        Processes input sentences, tokenizes them, and applies the model to get embeddings.
        """
        # for i, input_id in enumerate(input_ids):
        #     text = self.tokenizer.decode(input_id, skip_special_tokens=False)
        #     print(f"Sample {i} text:")
        #     print("input_id：", input_id)
        #     print("text：", text)
            
        with torch.no_grad():
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                with self.model.disable_adapter():
                    outputs = self.model(
                            input_ids, 
                            attention_mask=attention_mask, 
                            output_hidden_states=True, 
                            return_dict=True,
                            use_cache=False,
                            output_attentions=return_attentions
                    )
        hidden_states = outputs.hidden_states[-1].to(torch.float32)
        pooled_output = self._pool(input_ids, attention_mask, hidden_states)
        

        if return_attentions:
            # attention = outputs.attentions[-1]
            attention = outputs.attentions[-1].detach()
            for i in range(1, len(outputs.attentions)):
                attention += outputs.attentions[i]
            attention /= len(outputs.attentions)
            return pooled_output, attention
        else:
            return pooled_output
            
    def _pool(self, input_ids, attention_mask, last_hidden_states):
        seq_lengths = attention_mask.sum(dim=-1)
        embeddings = None
        # print("choose pooling,", self.pooling_mode)
        if self.pooling_mode == 'last':
            embeddings = last_hidden_states[:, -1, :]
        elif self.pooling_mode == 'first':
            embeddings = last_hidden_states[
                input_ids == self.tokenizer.bos_token_id
            ]
        elif self.pooling_mode == 'mean':
            embeddings = torch.stack(
                [
                    last_hidden_states[i, -length:, :].mean(dim=0)
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
            )
        elif self.pooling_mode == "last_eight_mean":
            embeddings = last_hidden_states[:, -8:, :]
            embeddings = embeddings.mean(dim=1)
           
        else:
            raise ValueError(f"Invalid pooling mode: {self.pooling_mode}")

        return embeddings
    
    
    
    def _prompt_and_tokenize(self, sentence):
        prompt = self.prompt
        tokenized_sentences = self.tokenizer.tokenize(sentence)
        truncated_sentence = tokenized_sentences[:self.text_max_length]
        reconstructed_texts = self.tokenizer.convert_tokens_to_string(truncated_sentence)
        prompt_sentence = prompt.replace('*sent_0*', reconstructed_texts).replace('_', ' ').strip()
        return prompt_sentence
    
    


   
    def length_prompt(self):
        
        prompt = self.prompt.replace("_", " ")
        left,right = prompt.split("*sent 0*")
        left_token = self.tokenizer.tokenize(left)
        right_token = self.tokenizer.tokenize(right)
        # print(f"left prompt length: {len(left_token)}, right prompt length: {len(right_token)}")
        return len(left_token), len(right_token)
    
    def remove_prompt(self, input_ids, attention):

        pad_token_id = self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)[0]
        prompt_left_length, prompt_right_length = self.length_prompt()
        non_pad_mask = (input_ids != pad_token_id)
        first_non_pad_indices = non_pad_mask.float().argmax(dim=1)
        batch_size, seq_len = input_ids.size()
        
        for i in range(batch_size):
                
            start = first_non_pad_indices[i]
            end = seq_len - prompt_right_length
            input_ids[i, start:start+prompt_left_length] = pad_token_id
            attention[i, start:start+prompt_left_length] = 0.0
            
            input_ids_right = input_ids[i, :end]
            input_ids_left_pad = torch.full((prompt_right_length,), pad_token_id, dtype=torch.long).to(input_ids.device)
            new_input_ids = torch.cat([input_ids_left_pad, input_ids_right], dim=0)
            input_ids[i]=new_input_ids
            
            attention_right = attention[i, :end]
            attention_left_pad = torch.zeros((prompt_right_length,)).to(attention.device)
            new_attention = torch.cat([attention_left_pad, attention_right], dim=0)
            attention[i]=new_attention
    
        return input_ids, attention
        
    def save_pretrained(self, save_directory: str):
        """
        Saves the model to the specified directory.
        """

        os.makedirs(save_directory, exist_ok=True)
        
        if hasattr(self, "module"):
            model_to_save = self.module     # 如果是 DDP 模式，获取实际的模型
        else:
            model_to_save = self            # 非 DDP 模式，直接保存当前模型

        model_to_save.model.save_pretrained(save_directory)
        config_path = os.path.join(save_directory, 'config.json')
        model_to_save.config.to_json_file(config_path)
        print(f"Model saved to {save_directory}")

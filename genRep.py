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
import torch.multiprocessing as mp
from torch import device, Tensor, nn
import numpy as np
from tqdm.autonotebook import tqdm, trange
from accelerate import Accelerator
from torch.amp import autocast
from peft.tuners.lora.layer import LoraLayer
import sys
import logging
# from models import MistralBiModel, LlamaBiModel, MistralBiForCausalLM, LlamaBiForCausalLM

def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

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
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, accelerator = None, logger=None, output_dir=None,
                 pooling_mode: str = "mean",
                 prompt: str = '*sent_0*', max_length: int = 256):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.max_length = max_length
        self.text_max_length = max_length - 10   #prompt length is  10
        self.prompt = prompt
        self.pooling_mode = pooling_mode
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
        trainable: bool = False,
        enable_bidirectional: bool = False,
        follow_mntp: bool = False,
        follow_llara: bool = False,
        **kwargs,
    ):

        keys = ["pooling_mode", "max_length", "prompt", "output_dir"]
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
            # config.gradient_checkpointing = True
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
                    checkpoint_path,
                )
                peft_target = peft_target.merge_and_unload()
                if hasattr(peft_target, 'peft_config'):
                    del peft_target.peft_config
                base_model.set_model_for_peft(peft_target)

            if follow_llara:
                peft_target = PeftModel.from_pretrained(
                    base_model,
                    checkpoint_path,
                )
                base_model = peft_target.merge_and_unload()
                
            lora_config = LoraConfig(
                r=lora_config['r'],                             
                lora_alpha=lora_config['lora_alpha'],           
                # target_modules=["q_proj", "v_proj"],
                target_modules=["q_proj", "v_proj", "k_proj","o_proj","gate_proj","up_proj","down_proj"],
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
                torch_dtype=torch.bfloat16 
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
                base_model = peft_target.merge_and_unload()
                
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
                return_attentions=False, **kwargs):
        """
        Processes input sentences, tokenizes them, and applies the model to get embeddings.
        """
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # for i, input_id in enumerate(input_ids):
            #     text = self.tokenizer.decode(input_id, skip_special_tokens=False)
            #     print(f"Sample {i} text:")
            #     print("input_id：", input_id)
            #     print("text：", text)
            outputs = self.model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True, 
                    return_dict=True,
                    use_cache=False,
                    output_attentions=return_attentions
            )
            pooled_output = self._pool(input_ids, attention_mask, outputs.hidden_states[-1])
    
            if return_attentions:
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
    
    def _prompt_and_tokenize_batch(self, sentences):
        prompt = self.prompt
        tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in sentences]
        truncated_sentences = [tokens[:self.text_max_length] for tokens in tokenized_sentences]
        reconstructed_texts = [self.tokenizer.convert_tokens_to_string(tokens) for tokens in truncated_sentences]
        prompt_sentences = [
            prompt.replace('*sent_0*', text).replace('_', ' ').strip() 
            for text in reconstructed_texts
        ]
        return prompt_sentences
    
    
    def _convert_to_str(self, instruction, text):
        # print("instruction, length is ", len(self.tokenizer.tokenize(instruction)))
        # print(self.tokenizer.tokenize(instruction))
        # print(f"{instruction.strip()} {text}")
        return (
            f"{instruction.strip()} {text}"
            if instruction
            else f"{text}"
        )
        
    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either a string (which means a single text)
        a list of ints (which means a single tokenized text), or a tuple of list of ints
        (representing several text inputs to the model).
        """
        if (
            isinstance(text, str)
            or (isinstance(text, list) and isinstance(text[0], int))
            or len(text) == 0
        ):  # Single text, list of ints, or empty
            return len(text)
        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        else:
            return sum([len(t) for t in text])

    
    
    def _encode(
        self,
        sentences_batch,
        device: Optional[str] = None,
        multiprocessing=False,
    ):
        if multiprocessing:
            # multiprocessing only supports CUDA devices at this time, so we ignore the value of device
            # and use cuda:rank for the device
            rank = mp.current_process()._identity[0]
            if device is None and torch.cuda.is_available():
                device = f"cuda:{rank % torch.cuda.device_count()}"
                
        self.to(device)
        # logger = Logger()
        # logger.add_stream_handler()
        # logger.add_file_handler(self.output_dir, rank)
        # sys.stdout = StreamToLogger(logger, logging.INFO)
        # sys.stderr = StreamToLogger(logger, logging.ERROR)
    
        if self.pooling_mode == "last_eight_mean":
            prefix = '"'
            suffix = '", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>'
            prefix_ids = self.tokenizer(prefix, truncation=True, max_length=self.max_length, return_tensors=None)['input_ids']
            suffix_ids = self.tokenizer(suffix, truncation=True, max_length=self.max_length, return_tensors=None, add_special_tokens=False)['input_ids']

            passages_inputs = []
            for text in sentences_batch:  # 假设输入是texts列表
                inputs = self.tokenizer(text,
                                       truncation=True,
                                       max_length=self.max_length - len(prefix_ids) - len(suffix_ids),
                                       padding=False,
                                       return_tensors=None,
                                       add_special_tokens=False)
        
                
                passages_input_ids = prefix_ids + inputs['input_ids'] + suffix_ids
                passages_attention_mask = [1] * len(passages_input_ids)
                
                passages_inputs.append({
                    'input_ids': passages_input_ids,
                    'attention_mask': passages_attention_mask
                })

            tokenized_sentences = self.tokenizer.pad(
                passages_inputs,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
                
        else:
            tokenized_sentences = self.tokenizer(
                sentences_batch, 
                truncation=True, 
                padding=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )

        tokenized_sentences = batch_to_device(tokenized_sentences, device)
        
        with torch.no_grad():
            output = self.forward(
                input_ids=tokenized_sentences["input_ids"], 
                attention_mask=tokenized_sentences["attention_mask"]
            )
            output = output.detach().cpu()  
        return output
    
    def genRep_encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 64,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = False,
    ) -> np.ndarray:

        if isinstance(sentences[0], str) and isinstance(sentences[-1], int):
            sentences = [sentences]
        # required for MEDI version of MTEB
        if isinstance(sentences[0], str):
            sentences = [[""] + [sentence] for sentence in sentences]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        
        concatenated_input_texts = []
        for sentence in sentences:
            assert isinstance(sentence[0], str)
            assert isinstance(sentence[1], str)
            concatenated_input_texts.append(
                self._convert_to_str(sentence[0], sentence[1])
            )
        sentences = concatenated_input_texts

        self.eval()

        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        all_embeddings = []
        
        with torch.no_grad(): 
            if torch.cuda.device_count() > 1:
                # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
                num_proc = torch.cuda.device_count()
                cuda_compatible_multiprocess = mp.get_context("spawn")
                with cuda_compatible_multiprocess.Pool(num_proc) as p:
                    batches = [
                        sentences_sorted[i:i + batch_size] 
                        for i in range(0, len(sentences_sorted), batch_size)
                    ]
                    progress_bar = tqdm(
                        total=len(batches),
                        desc="Batches",
                        disable=not show_progress_bar,
                    )
                    
                    outputs = []

                    def update(*args):
                        progress_bar.update()
                        
                    for batch in batches:
                        outputs.append(
                            p.apply_async(
                                self._encode,
                                args=(batch, None, True),
                                callback=update,
                            )
                        )
                    all_embeddings = [output.get() for output in outputs]
                    progress_bar.close()
                
            else:
                self.to(device)
                for start_index in trange(
                    0,
                    len(sentences),
                    batch_size,
                    desc="Batches",
                    disable=not show_progress_bar,
                ):
                    sentences_batch = sentences_sorted[
                        start_index : start_index + batch_size
                    ]
                    embeddings = self._encode(
                        sentences_batch, 
                        device=device, 
                    )
                    all_embeddings.append(embeddings)    
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]
        all_embeddings = all_embeddings.to(torch.float32)
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        print("return all_embeddings, length is ", all_embeddings.shape)
        return all_embeddings

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

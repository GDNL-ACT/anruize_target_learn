import os
import json
import torch
import torch.nn as nn
from genRep import GenRep
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments, get_scheduler, get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import default_data_collator
from transformers import TrainerCallback
from aa_dataset import WikiDataset, DataLoader
from torch.nn.functional import softmax, log_softmax
import argparse
from accelerate import Accelerator
from datetime import datetime
import torch.distributed as dist
import logging
from tqdm import tqdm
import pickle
import sys
from torch.utils.data import DistributedSampler
from itertools import islice

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, message):
        # 避免重复打印空行
        if message != "\n":
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass  # Required for compatibility
        
class Logger(logging.Logger):
    def __init__(self):
        super().__init__("target")

    def add_stream_handler(self):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
        self.addHandler(sh)

    def add_file_handler(self, save_dir):
        rank = dist.get_rank()
        fh = logging.FileHandler(save_dir + f"/log_rank{rank}.txt", "w")
        fh.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
        self.addHandler(fh)

    def set_verbosity_info(self):
        self.setLevel(logging.INFO)

    def set_verbosity_error(self):
        self.setLevel(logging.ERROR)
        
class Text2Target:
    def __init__(self, mode, cache_path, model, tokenizer, logger, 
                 tau=0.00001, accelerator = None, enable_bidirectional=False, follow_mntp=False):
        self.mode = mode
        self.cache_path = cache_path
        self.model = model
        self.tokenizer = tokenizer
        self.theta_random = None
        self.rank_random = None
        self.logger = logger
        self.tau = tau
        self.enable_bidirectional = enable_bidirectional
        self.follow_mntp = follow_mntp
        self.accelerator = accelerator or Accelerator()
        self.first_batch = True

    def rank_descending_tensor(self, stensor):
        sorted_indices = stensor.argsort(descending=True)  # 降序排序后的索引
        ranks = torch.zeros_like(stensor, dtype=torch.long)
        rank_values = torch.arange(0, stensor.size(0), device=stensor.device)
        ranks.scatter_(dim=0, index=sorted_indices, src=rank_values)
        return ranks
        
    def _load_theta_random(self, dataloader=None):
        
        if self.follow_mntp:
            self.enable_bidirectional = True
            random_path = os.path.join(self.cache_path, f"follow_mntp_bidirectional_theta_random.pkl")
        else:
            if self.enable_bidirectional:
                random_path = os.path.join(self.cache_path, f"bidirectional_theta_random.pkl")
            else:
                random_path = os.path.join(self.cache_path, f"theta_random.pkl")
            
        if os.path.exists(random_path):
            self.logger.info("Using precomputed theta_random.")
            with open(random_path, "rb") as f:
                self.logger.info(f"Loading cached theta_random from {random_path} ...")
                self.theta_random = pickle.load(f)
                self.rank_random = self.rank_descending_tensor(self.theta_random)   
                return self.theta_random
        else:
            self.theta_random = self.compute_theta_random(dataloader)
            self.rank_random = self.rank_descending_tensor(self.theta_random)   
            return self.theta_random
        
   
    def compute_theta_random(self, dataloader):

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        self.model.eval()  # 设为评估模式
        total_logits = None
        count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing Batches", total=len(dataloader)):
                device = next(self.model.parameters()).device
                batch = {k: v.to(device) for k, v in batch.items()}
                # prompt_lastpool_lmhead and noprompt_meanpool_lmhead
                if "lmhead" in self.mode or "llara" in self.mode:     
                    
                    outputs = self.model.forward_freeze(**batch)

                    model = self.model.module.model if hasattr(self.model, "module") else self.model.model

                    tokenizer = self.model.module.tokenizer if hasattr(self.model, "module") else self.model.tokenizer

                    model.resize_token_embeddings(len(tokenizer))
                    lm_head_dtype = model.lm_head.weight.dtype
                    outputs = outputs.to(lm_head_dtype)
                    
                    outputs = outputs.to(torch.float32)
                    logits = model.lm_head(outputs).contiguous().squeeze().float()


                elif self.mode == "prompt_attention":  
                    
                    _, attentions = self.model.forward_freeze(**batch, return_attentions=True)
                    tokenizer = self.model.module.tokenizer if hasattr(self.model, "module") else self.model.tokenizer

                    eos_attention = attentions[:, :, -1, :]  # 选取 " 位置的注意力 (B, H, S)
                    eos_attention = eos_attention.sum(dim=1)  # 在注意力头上求均值 (B, S)

                    input_ids_flat = batch["input_ids"].clone()
                    input_ids_flat, eos_attention = self.model.remove_prompt(input_ids_flat, eos_attention)

                    eos_attention_vocab = torch.zeros((eos_attention.shape[0], len(tokenizer))).to(eos_attention.device)
                    eos_attention_vocab = eos_attention_vocab.scatter_add_(1, input_ids_flat, eos_attention)

                    b = 0.015  # BM25 平滑超参数（可调）
                    logits = eos_attention_vocab / (b + eos_attention_vocab)
                    
                    del eos_attention
                    del eos_attention_vocab
                    torch.cuda.empty_cache()

                if total_logits is None:
                    total_logits = torch.zeros_like(logits, device=self.accelerator.device)
                total_logits += logits
                count += logits.shape[0]
                self.logger.info(f"Rank{rank}: {count}")

        dist.barrier()  
        
        count_tensor = torch.tensor([count], device=self.accelerator.device)

        gathered_counts = [torch.zeros_like(count_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_counts, count_tensor)
        gathered_logits = [torch.zeros_like(total_logits) for _ in range(world_size)]
        dist.all_gather(gathered_logits, total_logits)

        dist.barrier()
        if rank != 0: 
            self.theta_random = torch.zeros_like(total_logits[0], device=self.accelerator.device)
        else:
            self.logger.info("calculate!!")
            total_logits_gathered = torch.cat(gathered_logits, dim=0)
            total_count = sum(t.item() for t in gathered_counts)
            self.theta_random = total_logits_gathered.sum(dim=0) / total_count  # 计算平均值
            self._save_theta_random()
            self.logger.info(f"Total count across all processes: {total_count}")
        dist.barrier()
        dist.broadcast(self.theta_random, src=0)
        self.logger.info(f"Rank{rank}: {self.theta_random.shape}")
        return self.theta_random

    def _save_theta_random(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        if self.follow_mntp:
            self.enable_bidirectional = True
            random_path = os.path.join(self.cache_path, f"follow_mntp_bidirectional_theta_random.pkl")
        else:
            if self.enable_bidirectional:
                random_path = os.path.join(self.cache_path, f"bidirectional_theta_random.pkl")
            else:
                random_path = os.path.join(self.cache_path, f"theta_random.pkl")

        with open(random_path, "wb") as f:
            pickle.dump(self.theta_random, f)
        self.logger.info(f"Saved theta_random to {self.cache_path}.")

    
    def compute_theta_contrastive(self, batch):

        batch = {k: v for k, v in batch.items()}        

        if "llara_first" in self.mode:
            targets = []
            with torch.no_grad():
                
                batch_unique_ids = batch.get("unique_id")
                batch_texts = batch.get("line_text")
                
                self.logger.info("llara_first")
                batch_size = batch['input_ids'].size(0)
                for unique_ids, text in zip(batch_unique_ids, batch_texts):

                    vocab_tensor = torch.zeros(len(self.tokenizer), device=batch['input_ids'].device)

                    if len(unique_ids) > 0:
                        vocab_tensor[unique_ids] = 1.0 / len(unique_ids)
                    targets.append(vocab_tensor)
                    if self.first_batch:
                        self.first_batch = False
                        unique_ids = unique_ids.detach()
                        unique_words = [self.tokenizer.decode([token_id]) for token_id in unique_ids]
                        self.logger.info(f"text: {text}")
                        for token_id, token_words in islice(zip(unique_ids, unique_words), 2):
                            self.logger.info(f"token_ids: {token_id} -> token: {token_words}")

            targets = torch.stack(targets).detach()    
            return targets, targets
        
        # prompt_lastpool_lmhead and noprompt_meanpool_lmhead
        if "lmhead" in self.mode or "llara_second" in self.mode or "llara_original_style" in self.mode:
            outputs = self.model.forward_freeze(**batch)
            
            model = self.model.module.model if hasattr(self.model, "module") else self.model.model
            tokenizer = self.model.module.tokenizer if hasattr(self.model, "module") else self.model.tokenizer

            
            model.resize_token_embeddings(len(tokenizer))
            lm_head_dtype = model.lm_head.weight.dtype
            outputs = outputs.to(lm_head_dtype)
            
            outputs = outputs.to(torch.float32)   
            logits = model.lm_head(outputs).contiguous().squeeze().float()
        

            
        elif self.mode == "prompt_attention":  
            _, attentions = self.model.forward_freeze(**batch, return_attentions=True)
            tokenizer = self.model.module.tokenizer if hasattr(self.model, "module") else self.model.tokenizer

            eos_attention = attentions[:, :, -1, :]  
            eos_attention = eos_attention.sum(dim=1)

            input_ids_flat = batch["input_ids"].clone()
            
            input_ids_flat, eos_attention = self.model.remove_prompt(input_ids_flat, eos_attention)
            
            eos_attention_vocab = torch.zeros((eos_attention.shape[0], len(tokenizer))).to(eos_attention.device)            
            eos_attention_vocab = eos_attention_vocab.scatter_add_(1, input_ids_flat, eos_attention)
           
            b = 0.015  # BM25 平滑超参数（可调）
            logits = eos_attention_vocab / (b + eos_attention_vocab)
            
            del eos_attention
            del eos_attention_vocab
            torch.cuda.empty_cache()
            
        
        # logits = logits / self.attention_tau
        logits = softmax(logits, dim=-1)

        theta_random = self.theta_random.to(logits.device) #theta_random 在生成时，也做了attention_tau运算
        theta_random = softmax(theta_random, dim=-1)
        
        p_contrastive = logits  / (logits + theta_random + 1e-8)      
        p_contrastive = softmax(p_contrastive / self.tau, dim=-1)

        return logits, p_contrastive

class MaxStepCallback(TrainerCallback):
    def __init__(self, max_train_steps, logger=None):
        self.max_train_steps = max_train_steps
        self.logger = logger

    def on_step_end(self, args, state, control, **kwargs):
        # self.logger.info(f"global_step: {state.global_step}")
        if state.global_step >= self.max_train_steps:
            if self.logger:
                self.logger.info(f"Reached step {state.global_step}, stopping training (limit: {self.max_train_steps})")
            control.should_training_stop = True
        return control
        
class GenRepTrainer(Trainer):
    def __init__(self, model, train_dataset, logger, args, mode, text2target, accelerator = None, **kwargs):
        super().__init__(model=model, 
                         train_dataset=train_dataset, 
                         args=args,
                         **kwargs)
        self.train_dataset = train_dataset 
        self.mode = mode
        self.logger = logger
        self.text2target = text2target
        self.accelerator = accelerator or Accelerator()
        self.first_batch = True
    
    def get_train_dataloader(self):
        # 你可以在这里自定义数据加载的方式
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.accelerator.num_processes > 1:
            train_sampler = DistributedSampler(self.train_dataset)
        else:
            train_sampler = RandomSampler(self.train_dataset)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.custom_collate_fn,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
    def custom_collate_fn(self, batch):
        base = default_data_collator([{k: v for k, v in item.items() if k != "line_text" and k != "unique_id"} for item in batch])
        base["line_text"] = [item["line_text"] for item in batch]
        base["unique_id"] = [item.get("unique_id", None) for item in batch]

        return base
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        # self.logger.info(f"compute_loss:{inputs}")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        texts = inputs["line_text"]
        
        if "lmhead" in self.mode or "llara" in self.mode:

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 检查是否是 DDP 包装的模型
            model = self.model.module.model if hasattr(self.model, "module") else self.model.model
            tokenizer = self.model.module.tokenizer if hasattr(self.model, "module") else self.model.tokenizer
    
            model.resize_token_embeddings(len(tokenizer))
            lm_head_dtype = model.lm_head.weight.dtype
            outputs = outputs.to(lm_head_dtype)
            
            outputs = outputs.to(torch.float32)
            
            logits = model.lm_head(outputs).contiguous().squeeze().float()
            
        elif self.mode == "prompt_attention":  
            _, attentions = model(input_ids=input_ids, attention_mask=attention_mask, return_attentions=True)
            
            tokenizer = self.model.module.tokenizer if hasattr(self.model, "module") else self.model.tokenizer

            eos_attention = attentions[:, :, -1, :]  
            eos_attention = eos_attention.sum(dim=1)

            def unwrap_model(model):
                while hasattr(model, "module"):
                    model = model.module
                return model
            inner_model = unwrap_model(model)

            input_ids_flat = input_ids.clone() 
            input_ids_flat, eos_attention = inner_model.remove_prompt(input_ids_flat, eos_attention)
            
            eos_attention_vocab = torch.zeros((eos_attention.shape[0], len(tokenizer))).to(eos_attention.device)
            eos_attention_vocab = eos_attention_vocab.scatter_add_(1, input_ids_flat, eos_attention)
           
            b = 0.015  # BM25 平滑超参数（可调）
            logits = eos_attention_vocab / (b + eos_attention_vocab)

            del eos_attention
            del eos_attention_vocab
            torch.cuda.empty_cache()
        
        original_targets, targets = self.text2target.compute_theta_contrastive(inputs)
        logits_log_softmax = log_softmax(logits, dim=1)
        
        def prob2top(prob):
            top_probs, top_indices = torch.topk(prob, 100)
            top_tokens = top_indices.tolist()
            if "llara_first" in self.mode:
                top_words = [tokenizer.decode([token_id]) for token_id in top_tokens]
                for token, prob in zip(top_words, top_probs):
                    self.logger.info(f"{[token]}: {prob}")
                    
            else:
                top_sorts = self.text2target.rank_random[top_tokens]
                top_words = [tokenizer.decode([token_id]) for token_id in top_tokens]
            
                for token, prob, sort in zip(top_words, top_probs, top_sorts):
                    self.logger.info(f"{[token]}: {prob}({sort})")
                
        if self.first_batch:
            self.first_batch = False
            for text, original_target, target, logit in islice(zip(texts, original_targets, targets, logits_log_softmax), 2):
                self.logger.info(f"text: {text}")
                self.logger.info(f"logit: {logit.shape}")
                self.logger.info(f"target: {target.shape}")
                self.logger.info(f"Logits Probs:")
                prob2top(logit)
                if "llara_first" not in self.mode:
                    self.logger.info(f"Original Targets Probs:")
                    prob2top(original_target)
                self.logger.info(f"Targets Probs:")
                prob2top(target)

        targets = targets.to(logits.device).float()
        kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        loss = kl_div_loss(logits_log_softmax, targets)
        
        # 获取 rank
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # 先把 loss 从各个 rank 传输到 rank 0
        loss_list = [torch.zeros_like(loss) for _ in range(world_size)]
        if torch.isnan(loss).any():
            self.logger.info(f"Rank {rank} computed NaN loss!")
        dist.all_gather(loss_list, loss)

        # 只有 rank 0 进行打印
        if rank == 0:
            for i, l in enumerate(loss_list):
                self.logger.info(f"Rank {i} - Loss: {l}")

        return (loss, outputs) if return_outputs else loss
        
    
        
    def save_model(self, output_dir=None, _internal_call=True):

        self.first_batch = True
        self.text2target.first_batch = True
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.accelerator.is_main_process:
            
            os.makedirs(output_dir, exist_ok=True)

            if hasattr(self.model, "module"):
                model_to_save = self.model.module  # 如果是 DDP，获取实际模型
            else:
                model_to_save = self.model  # 非 DDP，直接使用当前模型
                
            model_to_save.save_pretrained(output_dir)

        self.accelerator.wait_for_everyone()
        
if __name__ == "__main__":
    
    # Step 1: 使用 argparse 定义参数
    parser = argparse.ArgumentParser()

    # 模型参数
    parser.add_argument("--model_name", type=str, default="opt-350m", help="模型名称")
    parser.add_argument("--base_model_name_or_path", type=str, default="/home/incoming/wuzy/models--facebook--opt-350m", help="模型路径")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="模型路径")
    
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA 的秩")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA 的 alpha 参数")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA 的 dropout 参数")
    parser.add_argument("--target_modules", type=str, nargs="+", default=["q_proj", "v_proj"], help="LoRA 的目标模块")
    parser.add_argument("--bias", type=str, default="none", help="LoRA 的 bias 参数")
    parser.add_argument("--prompt", type=str, default='This_sentence_:_"*sent_0*"_means_in_one_word:"', help="提示词")
    parser.add_argument("--pooling_mode", type=str, default="last", help="池化模式")
    parser.add_argument("--max_length", type=int, default=128, help="输入最大长度")
    # 数据集参数
    parser.add_argument("--dataset_path", type=str, default="./data/wiki1m_for_simcse.txt", help="数据集文件路径")    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./logits_path", help="输出文件夹")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=256, help="")
    parser.add_argument("--global_batch_size", type=int, default=64, help="")
    parser.add_argument("--num_epochs", type=int, default=1, help="训练的 epochs 数量")
    parser.add_argument("--weight_decay", type=float, default=0.00, help="权重衰减")
    parser.add_argument("--save_steps", type=int, default=1, help="模型保存步数")
    parser.add_argument("--follow_mntp", action='store_true', help='')
    parser.add_argument("--follow_llara", action='store_true', help='')
    parser.add_argument("--enable_bidirectional", action='store_true', help='')
    
    parser.add_argument("--mode", type=str, default="noprompt_meanpool_lmhead")
    parser.add_argument("--tau", type=float, default=0.1, help=" ")
    
    parser.add_argument("--scheduler_type", type=str, default="none", choices=["linear", "cosine_with_restarts", "none"], help="Scheduler type to use during training.")
    parser.add_argument("--max_train_steps", type=int, default=2, help="")

    args = parser.parse_args()
    
    num_gpus = torch.cuda.device_count()
    args.global_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * num_gpus
    
    # Step 2: 设置存档路径
    output_dir = os.path.join(args.output_dir, args.model_name, args.mode, f"ml_{args.max_length}")
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y_%m_%d_%H")
    if args.follow_mntp:
        if args.enable_bidirectional: 
            experiment_name = f"follow_mntp_bidirectional_lr_{args.learning_rate}_tua_{args.tau}_{current_time}"
        else:
            experiment_name = f"follow_mntp_lr_{args.learning_rate}_tua_{args.tau}_{current_time}"
    else:
        if args.enable_bidirectional: 
            experiment_name = f"bidirectional_lr_{args.learning_rate}_tua_{args.tau}_{current_time}"
        else:
            experiment_name = f"lr_{args.learning_rate}_tua_{args.tau}_{current_time}"
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    model_dir = os.path.join(experiment_dir, "model")
    logs_dir = os.path.join(experiment_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision="bf16")
    logger = Logger()
    logger.add_stream_handler()
    logger.add_file_handler(logs_dir)
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    
    with open(os.path.join(experiment_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # 打印当前时间和所有参数
    def print_current_time_and_args(args):
        # 获取当前时间
        logger.info("参数列表:")
        for arg, value in vars(args).items():
            logger.info(f"{arg}: {value}")

    # 执行打印函数
    print_current_time_and_args(args)
    

    # Step 4: 加载模型、配置和数据集
    lora_config = {
        'r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'target_modules': args.target_modules,
        'lora_dropout': args.lora_dropout,
        'bias': args.bias
    }
    # Step 3: 初始化加速器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = GenRep.from_pretrained(
        base_model_path=args.base_model_name_or_path,
        checkpoint_path=args.checkpoint_path,
        lora_config=lora_config,
        trainable=True,
        enable_bidirectional=args.enable_bidirectional,
        follow_mntp=args.follow_mntp,
        follow_llara=args.follow_llara,
        prompt=args.prompt,
        pooling_mode=args.pooling_mode,
        max_length=args.max_length,
        accelerator=accelerator,
        logger=logger
    )
   
    tokenizer = model.tokenizer
    dataset = WikiDataset(
        file_path=args.dataset_path,
        model=model,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mode=args.mode,
        accelerator=accelerator
    )

    text2target = Text2Target(
        model=model,
        tokenizer=tokenizer,
        mode=args.mode,
        cache_path=output_dir,
        accelerator=accelerator,
        logger=logger,
        enable_bidirectional=args.enable_bidirectional,
        follow_mntp=args.follow_mntp,
        tau=args.tau
    )
    # Step 5: 将模型和数据加载到设备

    def custom_collate_fn(batch):
        base = default_data_collator([{k: v for k, v in item.items() if k != "line_text" and k != "unique_id"} for item in batch])
        # base["line_text"] = [item["line_text"] for item in batch]
        return base
    if "llara_first" in args.mode:      
        pass
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=args.per_device_train_batch_size, 
            collate_fn = custom_collate_fn
        )
        
        model, dataloader, dataset, text2target = accelerator.prepare(model, dataloader, dataset, text2target)
        
        text2target.theta_random = text2target._load_theta_random(dataloader)
    
    
    # gradient_accumulation_steps = args.batch_size // args.per_device_train_batch_size
    # Step 6: 定义训练参数
    training_args = TrainingArguments(
        output_dir=model_dir,
        logging_dir=logs_dir,
        logging_steps=1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    # Step 7: 使用 Trainer 训练模型
    trainer = GenRepTrainer(
        model=model,                         # 模型
        args=training_args,                  # 训练参数
        train_dataset=dataset,               # 训练数据集
        text2target=text2target,
        mode=args.mode,
        accelerator=accelerator,
        logger=logger,
        callbacks=[MaxStepCallback(max_train_steps=args.max_train_steps, logger=logger)]
    )

    # num_training_steps = len(dataset) // args.per_step_batch_size
    print(f"max_train_steps: {args.max_train_steps}")
    
    trainer.create_optimizer()
    
    # if args.scheduler_type == "linear":
    #     scheduler = get_scheduler(
    #         name="linear",
    #         optimizer=trainer.optimizer,
    #         num_warmup_steps=0,
    #         num_training_steps=num_training_steps,
    #         # lr_scheduler_kwargs={"end_learning_rate": args.end_learning_rate}
    #     )
    #     trainer.lr_scheduler = scheduler
    # elif args.scheduler_type == "cosine_with_restarts":
    #     scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    #         optimizer=trainer.optimizer,
    #         num_warmup_steps=0,
    #         num_training_steps=num_training_steps,
    #         num_cycles=args.num_cycles
    #     )
        # trainer.lr_scheduler = scheduler
    
    logger.info("start training!")
    trainer.train()

    # 保存训练指标
    metrics = trainer.state.log_history
    with open(os.path.join(experiment_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
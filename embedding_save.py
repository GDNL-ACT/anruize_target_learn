import torch

from tqdm import tqdm
from genRep import GenRep
from aa_dataset import WikiDataset
from torch.utils.data import DataLoader
import pickle
import os
from pathlib import Path
import logging
import argparse
import sys
import torch.distributed as dist
from accelerate import Accelerator
from transformers import AutoTokenizer
import torch.distributed as dist
from transformers import default_data_collator
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice

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


def draw_figure(data, path):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 6))
    sns.lineplot(x=range(len(data)), y=data)
    plt.xlabel('$i$ (Dismention)')
    plt.ylabel('$v_i$ (Variation)')
    plt.title('Contribution to the aligned tokens')
    plt.savefig(path, bbox_inches='tight', dpi=300)

    
def get_embeddings(args, orignal_model:bool, eigen_matrix=None, logs_dir="./get_embedding_logs"):
    if orignal_model:
        output_path = os.path.join("./learn_from_target", args.model_name, args.mode, f"ml_{args.max_length}")
        logs_dir = os.path.join(output_path, "original_model_logs")
    else:
        output_path = args.checkpoint_path

    accelerator = Accelerator(mixed_precision="bf16")
    
    # logs_dir = os.path.join(output_path, "get_embedding_logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger = Logger()
    logger.add_stream_handler()
    logger.add_file_handler(logs_dir)
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    
    # embedding_path = os.path.join(output_path, "wiki_embeddings.pkl")
    
    file_path = os.path.join(output_path, f"embeddings.pkl")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            all_embeddings = pickle.load(f)    
  
    else:
        if orignal_model:
            model = GenRep.from_pretrained(
                base_model_path=args.base_model_name_or_path,
                checkpoint_path=None,
                enable_bidirectional=False,
                follow_mntp=False,
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
    
    
        else:
            model = GenRep.from_pretrained(
                base_model_path=args.base_model_name_or_path,
                checkpoint_path=args.checkpoint_path,
                checkpoint_path_mntp=args.checkpoint_path_mntp,
                enable_bidirectional=args.enable_bidirectional,
                follow_mntp=args.follow_mntp,
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
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(1024)))

        def custom_collate_fn(batch):
            base = default_data_collator([{k: v for k, v in item.items() if k != "line_text" and k != "unique_id"} for item in batch])
            base["line_text"] = [item["line_text"] for item in batch]
            return base
        
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn = custom_collate_fn)
    
        model.eval()  # 设置为评估模式
        model, dataloader = accelerator.prepare(model, dataloader)
        
        all_embeddings = []
        with torch.no_grad():
            first_batch = True
            for batch in tqdm(dataloader, desc="Running inference"):
        
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                texts = batch["line_text"]
                
                embeddings = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                gather_embeddings = accelerator.gather(embeddings).cpu()  # 多卡合并 + 移动到 CPU

                if orignal_model:
                    lm_head_dtype = model.module.model.lm_head.weight.dtype
                    embeddings = embeddings.to(lm_head_dtype)     
                    targets = model.module.model.lm_head(embeddings).contiguous().squeeze().float()
                    
                else:
                    lm_head_dtype = model.model.lm_head.weight.dtype
                    embeddings = embeddings.to(lm_head_dtype)     
                    targets = model.model.lm_head(embeddings).contiguous().squeeze().float()
                    
                def prob2top(prob):
                    top_probs, top_indices = torch.topk(prob, 30)
                    top_tokens = top_indices.tolist()
                    
                    top_words = [tokenizer.decode([token_id]) for token_id in top_tokens]
                    for token, prob in zip(top_words, top_probs):
                        logger.info(f"{[token]}: {prob}")
                if first_batch:   
                    for text, target in islice(zip(texts, targets), 20):
                        logger.info(f"text: {text}")
                        logger.info("Original_Targets Probs:")
                        prob2top(target)
                        first_batch = False

                
                all_embeddings.append(gather_embeddings)
                
        
        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings, dim=0)
        else:
            all_embeddings = torch.empty(0)    

    
    all_embeddings.cuda()
    logger.info(all_embeddings.shape)
    if orignal_model:
        U, S, V = torch.svd(all_embeddings)
        original_coeff = all_embeddings @ V
        if accelerator.is_main_process:
            file_path = os.path.join(output_path, f"embeddings.pkl")
            if not os.path.exists(file_path):
                file = open(file_path, "wb")
                pickle.dump(all_embeddings, file)
        
        
        return V, original_coeff
       
    else:
    
        trained_coeff = all_embeddings @ eigen_matrix
        if accelerator.is_main_process:
            file_path = os.path.join(output_path, f"embeddings.pkl")
            if not os.path.exists(file_path):
                file = open(file_path, "wb")
                pickle.dump(all_embeddings, file)
                
    
        return eigen_matrix, trained_coeff
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="opt-350m", help="模型名称")
    parser.add_argument("--base_model_name_or_path", type=str, default="/home/incoming/wuzy/models--facebook--opt-350m", help="模型路径")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="模型路径")
    parser.add_argument("--checkpoint_path_mntp", type=str, default=None, help="模型路径")
    parser.add_argument("--mode", type=str, default="noprompt_meanpool_lmhead")
    parser.add_argument("--pooling_mode", type=str, default="last", help="池化模式")
    parser.add_argument("--max_length", type=int, default=128, help="输入最大长度")
    parser.add_argument("--prompt", type=str, default='This_sentence_:_"*sent_0*"_means_in_one_word:"', help="提示词")
    parser.add_argument("--follow_mntp", action='store_true', help='默认是第一次训练')
    parser.add_argument("--enable_bidirectional", action='store_true', help='默认是第一次训练')
    
    parser.add_argument("--batch_size", type=int, default=64, help="每步训练的批大小")
    parser.add_argument("--dataset_path", type=str, default="./data/wiki1m_for_simcse.txt", help="数据集文件路径")  


    args = parser.parse_args()

    last_folder_name = Path(args.checkpoint_path).name
    parent_path = str(Path(args.checkpoint_path).parent)
    parent_path = str(Path(parent_path).parent)
    
    logs_dir = os.path.join(parent_path, "eval_logs", f"logs_{last_folder_name}")

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
        
    V, original_coeff = get_embeddings(args, True)
    _, trained_coeff = get_embeddings(args, False, V, logs_dir)
    different_value = (trained_coeff - original_coeff).mean(dim=0).cpu().numpy()

    figure_path = os.path.join(parent_path, "variation", f"variation_{last_folder_name}.png")
    figure_dir = os.path.dirname(figure_path)

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir, exist_ok=True)
    draw_figure(different_value, figure_path)
    print(f"The figure has been plotted successfully and saved at {figure_path}")


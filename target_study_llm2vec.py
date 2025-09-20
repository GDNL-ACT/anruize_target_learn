from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import Tensor
import torch
from torch.utils.data import Dataset
import os
# os.environ["HF_HOME"]="./cache"
import pandas as pd
from peft import PeftModel

class WikiDataset(Dataset):
    def __init__(self, file_path: str, file_name="wiki", max_length=128, doc_num=1000):
        
        self.file_name = file_name
        self.max_length = max_length
        
        if "pretrain" in self.file_name:
            import datasets
            ds = datasets.load_dataset(
                file_path,
                split='train',
                streaming=True
            )
            if "input" not in ds.features:
                raise ValueError("Dataset missing required field: input")
            # lines = ds["input"]
            lines = []
            for i, sample in enumerate(ds):
                lines.append(sample["input"])
                if i + 1 >= doc_num:
                    break
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

        lines = lines[:doc_num]
        self.lines = [line for line in lines if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]

        
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


dataset = WikiDataset("./data/wiki1m_for_simcse.txt")
documents = dataset.lines

# documents = [
#     "The capital of China is Beijing.",
#     "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
# ]

model_name = "LLM2vec-unsup"
# model_path = "/data/LLM/mistral/mistral-7b-v0_2-chat"


peft_path1 = "llm2vec_mntp_mistral_w_bidirection"
peft_path2 = "llm2vec_mntp_simcse_mistral_w_bidirection"

# model = AutoModelForCausalLM.from_pretrained(model_path)
from llm2vec import LLM2Vec
model = LLM2Vec.from_pretrained(
    peft_path1,
    peft_model_name_or_path=peft_path2,
    enable_bidirectional=True,      
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16
)
tokenizer = model.tokenizer


model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)

max_length = 128
batch_size = 2

save_path = f"./target_study/{model_name}/doc2logits.pt"
parent_dir = os.path.dirname(save_path)
os.makedirs(parent_dir, exist_ok=True)

# if os.path.exists(save_path):
#     doc2logits = torch.load(save_path)
# else:
all_logits = []
rows = []
hit_num = 0
with torch.no_grad():
    from tqdm import tqdm
    for start in tqdm(range(0, len(documents), batch_size), desc="Processing"):
        batch_docs = documents[start:start+batch_size]
        batch_dict = tokenizer(
            batch_docs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        
        # outputs = model(**batch_dict, output_hidden_states=True)
        # embeddings = last_token_pool(outputs.hidden_states[-1], batch_dict['attention_mask'])
        embeddings = model.encode(batch_docs)
        logits = model.lm_head(embeddings)
    
    
        top_probs, top_indices = torch.topk(logits, 10, dim=-1)

        input_ids = batch_dict["input_ids"]
        attention_mask = batch_dict["attention_mask"]
        hit_row = []
        for i in range(input_ids.shape[0]):
            
            sentence_ids = set(input_ids[i, attention_mask[i] == 1].tolist())
            top_list = top_indices[i].tolist()  
            for tid in top_list:
                if tid in sentence_ids:
                    hit_row.append(tid)

            top_words = [tokenizer.decode([token_id]) for token_id in top_list]
            top_probs_ = top_probs[i].tolist()


            sentence = batch_docs[i]
            print(f"text: {sentence}")
            row = [sentence]
            for words, probs in zip(top_words, top_probs_):
                print(f"{[words]}: {probs}")
                row.extend([words, probs])
            rows.append(row)
            
        hit_num += len(hit_row)
            
        logits = logits.detach().half()
        all_logits.append(logits.cpu())
    
    token_distribution = torch.cat(all_logits, dim=0).numpy()
    
doc2logits = {
    doc: token_distribution[i]
    for i, doc in enumerate(documents)
}


columns = ["text"]
for k in range(10):
    columns.extend([f"top_words[{k}]", f"top_probs_[{k}]"])
df = pd.DataFrame(rows, columns=columns)
output_path = os.path.join(parent_dir, "result.xlsx")
df.to_excel(output_path, index=False)
print(f"结果已保存到: {output_path}")

print(f"hit_words: {hit_num}")
with open(os.path.join(parent_dir, "hit_num.txt"), "w", encoding="utf-8") as f:
    f.write(str(hit_num))
torch.save(doc2logits, save_path)



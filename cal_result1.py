import os
import json
import pandas as pd
from tqdm.autonotebook import tqdm
# 配置范围
# pooling_method = ['last']  # 设置 last 的取值范围
# learning_rate = 5e-05  # 固定学习率
# max_length = 64  # 固定 max_length
# temp = 0.1
# task_name_list=[
#     "NFCorpus",
#     "SciFact",
#     "FiQA2018",
# ]

# id_range = list(range(10, 75, 5))  # 设置 id 的取值范围

# 基础路径
# base_path = "/home/LAB/anruize24/gene/results/mistral-7b"

# # 遍历 last_range 和 id_range
# for pooling in pooling_method:
#     # 初始化 Excel 数据存储结构
#     excel_data = {}

#     # 构建子目录路径
#     # sub_dir = f"ml_{max_length}_tp_{temp}_followOld_False_2025_02_02_22:25:00"
#     sub_dir = f"ml_{max_length}_tp_{temp}_followOld_True_2025_02_02_23:55:09"
#     sub_path = os.path.join(base_path, sub_dir)
#     print(sub_path)
     
#     for task_name in task_name_list:
         
#         for checkpoint_id in tqdm(id_range):
#             # 构建 JSON 文件路径
#             json_path = os.path.join(
#                 sub_path,
#                 f"model/checkpoint-{checkpoint_id}/no_model_name_available/no_revision_available/{task_name}.json"
#             )
#             print(json_path)
#             # 检查文件是否存在
#             if os.path.exists(json_path):
#                 # 读取 JSON 文件内容
#                 with open(json_path, 'r') as file:
#                     data = json.load(file)

#                 # 获取 task_name 和 main_score
#                 main_score = data.get("scores", {}).get("test", [{}])[0].get("main_score")

#                 # 将数据存入 Excel 表格结构
#                 if task_name not in excel_data:
#                     excel_data[task_name] = {}
#                 excel_data[task_name][f"checkpoint-{checkpoint_id}"] = main_score

#     # 将结果保存为 Excel
#     excel_file_path = os.path.join(sub_path, sub_dir+"_results.xlsx")
#     df = pd.DataFrame(excel_data)
#     df.index.name = "Checkpoint"
#     df.to_excel(excel_file_path)

#     print(f"Results saved to {excel_file_path}")


import os
import json
import pandas as pd
# noprompt_meanpool_lmhead

# 设置你的root路径
model = "mistral-7b"
method = "llara_second"
root_path = f"./learn_from_target/{model}/{method}"  # 替换为实际路径
max_length = 128
ml_file = f"ml_{max_length}"
sub_file = "lr_5e-05_tua_0.1_2025_07_18_16"
use_instructions = False

# 用于存储结果
results = []

# 遍历model文件夹下的所有子文件夹
model_path = os.path.join(root_path, ml_file, sub_file, "model")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型路径不存在: {model_path}")
for folder_name in os.listdir(model_path):
    folder_path = os.path.join(model_path, folder_name)
    if os.path.isdir(folder_path) and folder_name.startswith("checkpoint-"):
        if use_instructions:
            folder_path = os.path.join(model_path, "use_instructions")
        json_path = os.path.join(folder_path, "no_model_name_available", "no_revision_available", "SciFact.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                main_score = data.get("scores", {}).get("test", [{}])[0].get("main_score")
                results.append((folder_name, main_score))
            except Exception as e:
                print(f"读取失败：{json_path}，错误：{e}")

# 将结果写入Excel
results.sort(key=lambda x: int(x[0].split("-")[-1]))

df = pd.DataFrame(results, columns=["Checkpoint", "Main Score"])
sub_dir = model + "_" + method + "_" +ml_file + "_" + sub_file
if use_instructions:
    output_path = os.path.join(root_path, ml_file, sub_file, sub_dir + "_main_scores_use_instructions.xlsx")
output_path = os.path.join(root_path, ml_file, sub_file, sub_dir + "_main_scores.xlsx")
df.to_excel(output_path, index=False)

print(f"共记录 {len(results)} 个 checkpoint，结果已写入：{output_path}")


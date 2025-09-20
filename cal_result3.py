import os
import json
from openpyxl import Workbook
import pandas as pd

def extract_to_excel(folder_path, output_excel="output.xlsx"):
    # 任务顺序
    task_order = [
        # Classification
        "Banking77Classification","ImdbClassification","MTOPDomainClassification",
        "MassiveIntentClassification","MassiveScenarioClassification",
        "ToxicConversationsClassification","TweetSentimentExtractionClassification",
        "AmazonCounterfactualClassification",
        # Clustering
        "ArXivHierarchicalClusteringP2P","ArXivHierarchicalClusteringS2S",
        "BiorxivClusteringP2P.v2","MedrxivClusteringP2P.v2",
        "MedrxivClusteringS2S.v2","StackExchangeClustering.v2",
        "StackExchangeClusteringP2P.v2","TwentyNewsgroupsClustering.v2",
        # PairClassification
        "SprintDuplicateQuestions","TwitterSemEval2015","TwitterURLCorpus",
        # Reranking
        "AskUbuntuDupQuestions","MindSmallReranking",
        # Retrieval
        "ArguAna","CQADupstackGamingRetrieval","CQADupstackUnixRetrieval",
        "ClimateFEVERHardNegatives","FEVERHardNegatives","FiQA2018",
        "HotpotQAHardNegatives","SCIDOCS","TRECCOVID","Touche2020Retrieval.v3",
        # STS
        "BIOSSES","SICK-R","STS12","STS13","STS14","STS15",
        "STSBenchmark","STS17","STS22.v2",
        # Summarization
        "SummEvalSummarization.v2"
    ]
    order_index = {name: i for i, name in enumerate(task_order)}

    # 1️⃣ 读取所有 json，收集 task_name 和 main_score
    records = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".json"):
            fpath = os.path.join(folder_path, filename)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                task_name = data.get("task_name")
                main_score = data.get("scores", {}).get("test", [{}])[0].get("main_score")
                if task_name is not None and main_score is not None:
                    records.append((task_name, main_score))
                else:
                    print(f"⚠️ 跳过 {filename}，缺少字段")
            except Exception as e:
                print(f"读取 {filename} 出错: {e}")

    # 排序
    df = pd.DataFrame(records, columns=["task_name", "main_score"])
    df["order"] = df["task_name"].map(order_index).fillna(len(order_index))
    df.sort_values("order", inplace=True)
    df.drop(columns="order", inplace=True)

    # 2️⃣ 映射 task_class
    task_to_class = {
        # ——映射同上，这里省略，保持与上一段一致——
        "Banking77Classification":"Classification","ImdbClassification":"Classification",
        "MTOPDomainClassification":"Classification","MassiveIntentClassification":"Classification",
        "MassiveScenarioClassification":"Classification","ToxicConversationsClassification":"Classification",
        "TweetSentimentExtractionClassification":"Classification","AmazonCounterfactualClassification":"Classification",

        "ArXivHierarchicalClusteringP2P":"Clustering","ArXivHierarchicalClusteringS2S":"Clustering",
        "BiorxivClusteringP2P.v2":"Clustering","MedrxivClusteringP2P.v2":"Clustering",
        "MedrxivClusteringS2S.v2":"Clustering","StackExchangeClustering.v2":"Clustering",
        "StackExchangeClusteringP2P.v2":"Clustering","TwentyNewsgroupsClustering.v2":"Clustering",

        "SprintDuplicateQuestions":"PairClassification","TwitterSemEval2015":"PairClassification",
        "TwitterURLCorpus":"PairClassification",

        "AskUbuntuDupQuestions":"Reranking","MindSmallReranking":"Reranking",

        "ArguAna":"Retrieval","CQADupstackGamingRetrieval":"Retrieval",
        "CQADupstackUnixRetrieval":"Retrieval","ClimateFEVERHardNegatives":"Retrieval",
        "FEVERHardNegatives":"Retrieval","FiQA2018":"Retrieval","HotpotQAHardNegatives":"Retrieval",
        "SCIDOCS":"Retrieval","TRECCOVID":"Retrieval","Touche2020Retrieval.v3":"Retrieval",

        "BIOSSES":"STS","SICK-R":"STS","STS12":"STS","STS13":"STS","STS14":"STS","STS15":"STS",
        "STSBenchmark":"STS","STS17":"STS","STS22.v2":"STS",

        "SummEvalSummarization.v2":"Summarization"
    }
    df["task_class"] = df["task_name"].map(task_to_class)

    # 3️⃣ 按类求平均
    class_scores = (df.groupby("task_class", dropna=True)["main_score"]
                      .mean().reset_index()
                      .rename(columns={"main_score": "class_score"}))

    # 4️⃣ 写入一个 Excel，两个 sheet
    with pd.ExcelWriter(output_excel) as writer:
        df.to_excel(writer, sheet_name="Results", index=False)
        class_scores.to_excel(writer, sheet_name="ClassScores", index=False)

    print(f"✅ 已生成 {output_excel}，包含两个 sheet：Results 和 ClassScores")
# 调用示例
if __name__ == "__main__":
    model = "llama3"
    method = "noprompt_meanpool_lmhead"
    stric = "bare"
    step="secondStep"
    # folder = f"./learn_target_pretrain/OriginModel/{model}/{method}/ml_128/{stric}/{step}/no_model_name_available/no_revision_available"  # 修改为你的文件夹路径
    # excel = folder + f"/aa_{model}_{method}_tasks_scores.xlsx"
    folder = f"llm2vec_llama3/no_model_name_available/no_revision_available"
    excel = folder + f"/aa_{model}_tasks_scores.xlsx"
    extract_to_excel(folder, excel)

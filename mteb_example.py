# import mteb
# # from sentence_transformers import SentenceTransformer

# # Define the sentence-transformers model name
# # model_name = "average_word_embeddings_komninos"

# model = mteb.get_model("sentence-transformers/llm2vec") # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)

# QUICK_EVAL = [
#     # # Classification
#     "Banking77Classification",
#     "EmotionClassification",
#     # # Clustering
#     "MedrxivClusteringS2S",
#     # PairClassification
#     "TwitterSemEval2015",
#     # # Reranking
#     "AskUbuntuDupQuestions",
#     # Retrieval
#     "ArguAna",
#     "NFCorpus",
#     "SciFact",
#     # STS
#     "BIOSSES",
#     "STS17",
#     "STSBenchmark",
#     # Summarization
#     "SummEval",
# ]



# QUICK_EVAL = [
#         # "NFCorpus",
#         "SciFact",
#         # "FiQA2018",
#     ]

# tasks = mteb.get_tasks(tasks=QUICK_EVAL)
# evaluation = mteb.MTEB(tasks=tasks)
# # evaluation = mteb.MTEB()
# results = evaluation.run(model, output_folder=f"results/downloadData")

from sentence_transformers import SentenceTransformer
import mteb
import torch
from mteb import MTOPDomainClassification
model_name = "sentence-transformers/all-MiniLM-L6-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mteb.get_model(model_name)

model.model.to(device)

# benchmark = mteb.get_benchmark("MTEB(eng, v2)")
# benchmark.tasks = [
#         task for task in benchmark.tasks
#         if not isinstance(task, MTOPDomainClassification)
#     ]
# QUICK_EVAL_2 = [
#     # # Classification
#     "Banking77Classification",
#     "MassiveScenarioClassification",
#     # # Clustering
#     "MedrxivClusteringS2S.v2",
#     # PairClassification
#     "TwitterSemEval2015",
#     # # Reranking
#     "AskUbuntuDupQuestions",
#     # Retrieval
#     "ArguAna",
#     "ClimateFEVERHardNegatives",
#     "FEVERHardNegatives",
#     # STS
#     "BIOSSES",
#     "STS17",
#     "STSBenchmark",
#     # Summarization
#     "SummEvalSummarization.v2",
# ]

benchmark = mteb.get_benchmark("MTEB(eng, v2)")
evaluation = mteb.MTEB(tasks=benchmark)

    
results = evaluation.run(model, output_folder=f"results/downloadData_20250601")



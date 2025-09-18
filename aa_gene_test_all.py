import argparse
from typing import Any, Union,Optional, List
import mteb
import json
import torch
import numpy as np
from util.instructions import task_to_instruction
from util.text_formatting_utils import corpus_to_texts
from genRep import GenRep
from tqdm import tqdm
import torch.multiprocessing as mp
import os
import sys
import torch.distributed as dist
import logging

# os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"      
def genRep_instruction(instruction):
    if len(instruction) > 0 and instruction[-1] != ":":
        instruction = instruction.strip(".") + ":"
    return instruction

class genRepWrapper:
    def __init__(self, model, task_to_instructions, mode, use_instructions:bool, batch_size = 64):
        self.batch_size = batch_size    
        self.task_to_instructions = task_to_instructions
        self.model = model
        self.use_instructions = use_instructions
        self.mode = mode
        
    def similarity(self, a: torch.Tensor, b: torch.Tensor):
        """Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        return torch.mm(a, b.transpose(0, 1))
    
    def encode(
        self,
        sentences: list[str],
        *,
        prompt_name: str = None,
        **kwargs: Any,  # noqa
    ) -> np.ndarray:
        print("in encode")
        print(prompt_name)
        print(self.use_instructions)
        print(kwargs)
        if prompt_name is not None and self.use_instructions:
            print("prompt name is ", prompt_name)
            instruction = (
                self.task_to_instructions[prompt_name]
                if self.task_to_instructions
                and prompt_name in self.task_to_instructions
                else genRep_instruction(task_to_instruction(prompt_name))
            )
            print("instruction is ", instruction)
        else:
            print("no instruction")
            instruction = ""

        if "last" in self.mode or "attention" in self.mode:
            prompt_sentences = self.model._prompt_and_tokenize_batch(sentences)
        else:
            prompt_sentences = sentences
            
        flattened_sentences = [[instruction, sentence] for sentence in prompt_sentences]
        # print(flattened_sentences[0])
        return self.model.genRep_encode(flattened_sentences, self.batch_size)

    def encode_corpus(
        self,
        corpus: Union[list[dict[str, str]], dict[str, list[str]], list[str]],
        prompt_name: str = None,
        **kwargs: Any,
    ) -> np.ndarray:
        print("in encode_corpus")
        sentences = corpus_to_texts(corpus, sep=" ")
        if "request_qid" in kwargs:
            kwargs.pop("request_qid")
        
        if "last" in self.mode or "attention" in self.mode:
            prompt_sentences = self.model._prompt_and_tokenize_batch(sentences)
        else:
            prompt_sentences = sentences
            
        flattened_sentences = [["", sentence] for sentence in prompt_sentences]
        
        return self.model.genRep_encode(flattened_sentences, self.batch_size)

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        print("in encode_queries")
        return self.encode(queries, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3-8b",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default="/home/incoming/LLM/llama3/llama3-8b-instruct",
    )
    parser.add_argument(
        "--peft_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pre_checkpoint_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pre_checkpoint_path1",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pre_checkpoint_path2",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default='This_sentence_:_"*sent_0*"_means_in_one_word:"',
    )
    parser.add_argument(
        "--task_to_instructions_fp",
        type=str,
        default="./test_config/mteb/task_to_instructions.json",
    )
    parser.add_argument(
        "--use_instructions", action="store_true", help="Enable instructions mode"
    )
    parser.add_argument("--enable_bidirectional", action='store_true', help='')
    # parser.add_argument("--enable_causal", action='store_true', help='')
    parser.add_argument("--follow_mntp", action='store_true', help='')
    parser.add_argument("--follow_llara", action='store_true', help='')
    parser.add_argument("--follow_llara_double", action='store_true', help='')
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--pooling_mode",
        type=str,
        default="mean",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="noprompt_meanpool_lmhead",
    )
    parser.add_argument("--output_dir", type=str, default="/")
    args = parser.parse_args()

    os.chdir("/home/anruize24/gene")
    # if not os.path.exists(os.getcwd()):
    #     os.chdir("/home/anruize24") 

    print("Current working directory:", os.getcwd())

    
    task_to_instructions = None
    if args.task_to_instructions_fp is not None:
        with open(args.task_to_instructions_fp, "r") as f:
            task_to_instructions = json.load(f)

    genRep_model = GenRep.from_pretrained(
           base_model_path = args.base_model_name_or_path, 
           pre_checkpoint_path=args.pre_checkpoint_path, 
           pre_checkpoint_path1=args.pre_checkpoint_path1, 
           pre_checkpoint_path2=args.pre_checkpoint_path2, 
           checkpoint_path=args.peft_model_name_or_path, 
           prompt = args.prompt,
           enable_bidirectional=args.enable_bidirectional,
           follow_mntp=args.follow_mntp,
           follow_llara=args.follow_llara,
           follow_llara_double=args.follow_llara_double,
           pooling_mode = args.pooling_mode,
           max_length = args.max_length,
           device_map="cuda" if torch.cuda.is_available() else "cpu",
           torch_dtype=torch.bfloat16,
           output_dir=args.output_dir,
    )

    print("here: ", args.use_instructions)
    model = genRepWrapper(model = genRep_model, 
                          mode = args.mode,
                          task_to_instructions=task_to_instructions, 
                          use_instructions= args.use_instructions, 
                          batch_size = args.batch_size
                         )

        
    benchmark = mteb.get_benchmark("MTEB(eng, v2)")
    evaluation = mteb.MTEB(tasks=benchmark)
    
    results = evaluation.run(model, output_folder=args.output_dir, 
                             encode_kwargs={"batch_size": args.batch_size})

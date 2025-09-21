import torch.multiprocessing as mp
import numpy as np
import torch
from typing import Any, Union,Optional, List
from torch import device, Tensor
from tqdm.autonotebook import tqdm, trange
class genRepEncode:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


    def _prompt_and_tokenize_batch(self, sentences):
        prompt = self.prompt
        tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in sentences]
        truncated_sentences = [tokens[:self.model.text_max_length] for tokens in tokenized_sentences]
        reconstructed_texts = [self.tokenizer.convert_tokens_to_string(tokens) for tokens in truncated_sentences]
        prompt_sentences = [
            prompt.replace('*sent_0*', text).replace('_', ' ').strip() 
            for text in reconstructed_texts
        ]
        return prompt_sentences
        
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

    def _convert_to_str(self, instruction, text):
        # print("instruction, length is ", len(self.tokenizer.tokenize(instruction)))
        # print(self.tokenizer.tokenize(instruction))
        # print(f"{text}")
        return (
            f"{instruction.strip()} {text}"
            if instruction
            else f"{text}"
        )
        
    def batch_to_device(self, batch, target_device: device):
        """
        send a pytorch batch to a device (CPU/GPU)
        """
        for key in batch:
            if isinstance(batch[key], Tensor):
                batch[key] = batch[key].to(target_device)
        return batch


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
                
        self.model.to(device)
        self.model.eval()
    
        tokenized_sentences = self.tokenizer(
            sentences_batch, 
            truncation=True, 
            padding=True, 
            max_length=self.model.max_length, 
            return_tensors="pt"
        )

        tokenized_sentences = self.batch_to_device(tokenized_sentences, device)
        
        with torch.no_grad():
            output = self.model(
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
        

        concatenated_input_texts = []

        for sentence in sentences:
            assert isinstance(sentence[0], str)
            assert isinstance(sentence[1], str)
            concatenated_input_texts.append(
                self._convert_to_str(sentence[0], sentence[1])
            )
        sentences = concatenated_input_texts 

        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        all_embeddings = []

        with torch.no_grad(): 
            if torch.cuda.device_count() > 1:
                
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
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # self.model.to(device)
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
        return all_embeddings

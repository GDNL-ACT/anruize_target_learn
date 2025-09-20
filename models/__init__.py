from .bidirectional_mistral import MistralBiModel, MistralBiForCausalLM
from .bidirectional_llama import LlamaBiModel, LlamaBiForCausalLM
# from .bidirectional_gemma import GemmaBiModel, GemmaBiForMNTP
# from .bidirectional_qwen2 import Qwen2BiModel, Qwen2BiForMNTP
def bidirectional_get_model_class(config_class_name):
    if config_class_name == "MistralConfig":
        return MistralBiForCausalLM
    elif config_class_name == "LlamaConfig":
        return LlamaBiForCausalLM
    else:
        raise ValueError(
            f"{config_class_name} is not supported yet with bidirectional models."
        )
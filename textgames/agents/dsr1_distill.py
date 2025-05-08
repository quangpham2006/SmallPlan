#%%
import os
import re

#%%
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from textgames import THE_GAMES, GAME_NAMES, LEVEL_IDS
from agents import run_with_agent

#%%
def set_all_seed(seed=42):
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#%%
def _getenv_as_int(attr, default=None):
    ret = os.getenv(attr, default)
    return None if ret is None else int(ret)


GAME_ST, GAME_ED = _getenv_as_int("TG_GAME_ST", None), _getenv_as_int("TG_GAME_ED", None)
LVL_ST, LVL_ED = _getenv_as_int("TG_LEVEL_ST", None), _getenv_as_int("TG_LEVEL_ED", '3')
SID_ST, SID_ED = _getenv_as_int("TG_SID_ST", None), _getenv_as_int("TG_SID_ED", None)
N_TURNS = _getenv_as_int("TG_N_TURNS", 1)
ONE_SHOT = bool(int(os.getenv("TG_ONESHOT", "0")))
MAX_NEW_TOKENS = _getenv_as_int("TG_MAX_NEW_TOKENS", 12000)
DSR1_SIZE = os.getenv("TG_DSR1_SIZE", "14")    # {1.5, 7, 8, 14, 32, 70}
DSR1_NAME = {
    "1.5": "Qwen-1.5",
    "7": "Qwen-7",
    "8": "Llama-8",
    "14": "Qwen-14",
    "32": "Qwen-32",
    "70": "Llama-70",
}


#%%
def dsr1_postproc(response_txt_batch, *args, **kwargs):
    response_txt_batch = [response_txt_batch]
    ret = []
    for response_txt in response_txt_batch:
        _match = None
        for pat in [
            re.compile(r'\\boxed\{([\s\S]*)}'),
            re.compile(r'</think>\n([\s\S]*)$'),
            re.compile(r'^```\n?([^`]*)\n?```'),
        ]:
            matches = pat.search(response_txt)
            if matches:
                _match = matches.group(1).strip()
                break
        if _match is not None:
            ret.append(_match)
        else:
            ret.append(response_txt[:256].strip() if response_txt else "")
    return ret[0]


#%%
def get_dsr1_response(texts_batch, *args, **kwargs):
    # global model, tokenizer
    texts_batch = [texts_batch]
    for texts in texts_batch:
        if len(texts) > 1 and texts[1].startswith('Correct guess.'):
            texts[1] = f"\\boxed{{{texts[1]}}}"
    messages = [
        [
            {"role": "user",
             "content": f"{text}\nPlease reason step by step, and put your final answer within \\boxed{{}} as plain text."}
            if i % 2 == 0 else
            {"role": "assistant", "content": {text}}
            for i, text in enumerate(texts)
        ]
        for texts in texts_batch
    ]
    text_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer(text_inputs, return_tensors="pt", add_special_tokens=False).to(model.device)
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = [
        _output_ids[len(input_ids):] for input_ids, _output_ids in zip(model_inputs.input_ids, output_ids)
    ]
    response = [r.strip() for r in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]
    return response[0]


#%%
# response = get_dsr1_response(texts)
# print(dsr1_postproc(response))


#%%
if __name__ == "__main__":
    fp_out = (f"model_outputs/__runs__/results_deepseek-r1-distill-{DSR1_SIZE}b"
              f"{'.1s' if ONE_SHOT else '.zs'}"
              f"{'' if GAME_ST is None else f'.{GAME_ST}'}"
              f"{'' if LVL_ST is None else f'.{LVL_ST}'}"
              f".jsonl")

    set_all_seed()
    model_name = f"deepseek-ai/DeepSeek-R1-Distill-{DSR1_NAME[DSR1_SIZE]}B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    model.generation_config.temperature = None
    model.generation_config.top_k = None
    model.generation_config.top_p = None

    run_with_agent(
        fp_out,
        get_dsr1_response,
        dsr1_postproc,
        n_turns=N_TURNS,
        game_names_list=GAME_NAMES[GAME_ST:GAME_ED],
        level_ids_list=LEVEL_IDS[LVL_ST:LVL_ED],
        sid_indices=(list(map(lambda r: f"session_{r:04}", range(SID_ST or 0, SID_ED or 1000)))
                     if SID_ST or SID_ED else None),
        prepend_example=ONE_SHOT,
        # remove_if_output_file_exist=False,
        assistant_uses_raw_response=False,
    )

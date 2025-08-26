#%%
import os
import re

#%%
import os
import json
import torch
import numpy as np
from pathlib import Path
from transformers import set_seed
from textgames import GAME_NAMES, LEVEL_IDS, game_filename
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
GPT_MODEL = os.getenv("TG_GPT_MODEL", "")
# MAX_NEW_TOKENS = _getenv_as_int("TG_MAX_NEW_TOKENS", 12000)


#%%
def preload_responses():
    responses_all = dict()
    for _turn in range(1, N_TURNS+1):
        fp = os.getenv(
            f"TG_GPT_OUTPUT_FILE_TURN_{_turn}",
            (f"model_outputs/__runs__/chatgpt_4o_mini_results/raw/batch_output_chatgpt-{GPT_MODEL}_turn{_turn}"
             f"{'.1s' if ONE_SHOT else '.zs'}.jsonl")
        )
        if not Path(fp).exists():
            if _turn < N_TURNS:
                print(f"  batch_output turn {_turn} is not available. path: \"{fp}\"")
            break
        with open(fp, "r", encoding="utf8") as i:
            data = [json.loads(line) for line in i]
        for d in data:
            sid, g = d['custom_id'].rsplit('-', 2)[-2:]
            msg = d['response']['body']['choices'][0]['message']
            responses_all.setdefault((g, _turn), dict())[sid] = msg['content']
            responses_all[g, _turn][sid] = msg['content']
            # assert msg['role'] == 'assistant'
            # assert msg['refusal'] is None
        # assert sum(len(v) for v in responses_all.values()) == 24000
    return responses_all
RESPONSES_ALL = preload_responses()
print(f"len(RESPONSES_ALL) = {len(RESPONSES_ALL)}")


#%%
def gpt_postproc(response_txt_batch, *args, **kwargs):
    response_txt_batch = [response_txt_batch]
    ret = []
    for response_txt in response_txt_batch:
        if response_txt is None:
            ret.append(response_txt)
            continue
        cur = None
        for pat in [
            re.compile(r'^```\n?([^`]*)\n?```'),
            # re.compile(r'\*\*\"?([^\"*]*)\"?\*\*'),
            re.compile(r'((.|\n)*)\n\nExplanation:\n'),
        ]:
            match = pat.search(response_txt)
            if match:
                cur = match.group(1).strip()
                # .replace(" ", "")
                break
        ret.append(cur if cur else response_txt)
    return ret[0]


#%%
def get_gpt_response(texts, game_name, difficulty_level, turn, *args, **kwargs):
    # global model, tokenizer
    sid = kwargs['sid']    # sid must be fed as params
    messages = [
        ({"role": "user", "content": text}
         if i % 2 == 0 else
         {"role": "assistant", "content": text})
        for i, text in enumerate(texts)
    ]

    response = None
    responses_all = RESPONSES_ALL.get((f"{game_filename(game_name)}_{difficulty_level}", turn), {})
    if responses_all:
        response = responses_all[sid]
    elif fp_next := os.getenv("TG_GPT_NEXTTURN_OUTPUT_FILE", None):
        with open(fp_next, "a", encoding="utf8") as o:
            o.write(json.dumps({
                'custom_id': f"{sid}-{game_filename(game_name)}_{difficulty_level}",
                "method": "POST", "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini-2024-07-18",
                    "max_completion_tokens": 200,
                    # "messages": [],
                    'messages': messages,
                    "seed": 42,
                    "temperature": 0,
                }
            }))
            o.write("\n")

    return response


#%%
if __name__ == "__main__":
    fp_out = (f"model_outputs/__runs__/chatgpt_4o_mini_results/process/results_chatgpt-{GPT_MODEL}"
              f"{'.1s' if ONE_SHOT else '.zs'}"
              f"{'' if GAME_ST is None else f'.{GAME_ST}'}"
              f"{'' if LVL_ST is None else f'.{LVL_ST}'}"
              f".jsonl")

    set_all_seed()

    run_with_agent(
        fp_out,
        get_gpt_response,
        gpt_postproc,
        n_turns=N_TURNS,
        game_names_list=GAME_NAMES[GAME_ST:GAME_ED],
        level_ids_list=LEVEL_IDS[LVL_ST:LVL_ED],
        sid_indices=(list(map(lambda r: f"session_{r:04}", range(SID_ST or 0, SID_ED or 1000)))
                     if SID_ST or SID_ED else None),
        prepend_example=ONE_SHOT,
        # remove_if_output_file_exist=False,
        assistant_uses_raw_response=False,
    )

#%%
import re
import random
from typing import List, Optional
from itertools import chain

from textgames.base_game import BaseGame
from textgames.assets.word_list import PrefixTrie, get_word_list_by_length

#%%
# len_count = dict(sorted([(k, len(v)) for k, v in WORDS_BY_LEN.items()]))
# cumsum, weighted = 0, 0
# for k, v in len_count.items():
#     cumsum += v
#     weighted += k*v
#
# #%%
# print(weighted / cumsum)


#%%
def find_solution(size, word_list):
    """Given the board size and the list of words, find a possible crossword arrangement."""
    prefix_trie = PrefixTrie(word_list)
    p_row, p_col = [prefix_trie.root] * size, [prefix_trie.root] * size

    def find_char_for(r, c):
        if r >= size:
            return True
        if c >= size:
            return find_char_for(r + 1, 0)

        cur_r, cur_c = p_row[r], p_col[c]
        edges = list(cur_r.children.keys())
        for ch in random.sample(edges, k=len(edges)):
            nex_r = cur_r.children[ch]
            if ch not in cur_c.children:
                continue
            nex_c = cur_c.children[ch]

            p_row[r], p_col[c] = nex_r, nex_c
            nex_r.capacity -= 1
            nex_c.capacity -= 1
            if (nex_r.capacity > -1) and (nex_c.capacity > -1) and find_char_for(r, c+1):
                return True
            nex_c.capacity += 1
            nex_r.capacity += 1
            p_row[r], p_col[c] = cur_r, cur_c

    if not find_char_for(0, 0):
        return None
    else:
        return [p.word for p in p_row], [p.word for p in p_col]


#%%


#%%
class CrosswordArrangerGame(BaseGame):
    @staticmethod
    def get_game_name() -> str:
        return "📰\tCrossword Arranger"

    def __init__(self, board_size: Optional[int] = None, full_word_list: Optional[List[str]] = None):
        super().__init__()
        self.board_size = board_size
        self.full_word_list = full_word_list
        self.possible_ans = None
        self.noise_ratio = None
        self.word_list = None
        self.exclude_states = ['full_word_list', 'possible_ans', 'noise_ratio']

    def _generate_new_game(self, *args, **kwargs) -> None:
        if kwargs.get("no_ans_prob", .0) > .0:
            raise NotImplementedError("Arranger with No Answer is not yet implemented")
        if not kwargs.get("no_duplicate", True):
            raise NotImplementedError("Arranger with Duplicate word is not yet implemented")

        self.board_size = int(kwargs.get("board_size", self.board_size or 3))
        self.full_word_list = kwargs.get("full_word_list", self.full_word_list or get_word_list_by_length(corpus=(
            {"oxford5k_opal"} if self.board_size < 5 else {"oxford5k_opal", "nltk_words"}
        ))[self.board_size])

        if ("preset_config" in kwargs) and (kwargs["preset_config"] == 1):
            # car
            # ago
            # bed
            min_word_list = ["age", "ago", "bed", "cab", "car", "rod"]
            self.possible_ans = ["car", "ago", "bed"]
            self.noise_ratio = .25

        else:
            ans_hor, ans_ver = find_solution(self.board_size, word_list=self.full_word_list)
            min_word_list = list(chain(ans_hor, ans_ver))
            self.possible_ans = ans_hor
            self.noise_ratio = kwargs.get("noise_ratio", self.noise_ratio or .5)

        self.word_list = [*min_word_list]
        for _ in range(round(len(min_word_list) * self.noise_ratio / (1 - self.noise_ratio))):
            while (next_word := random.choice(self.full_word_list)) in self.word_list:
                pass
            self.word_list.append(next_word)
        self.word_list = sorted(self.word_list)

    def _load_game(self, state_string):
        pat_board_size = re.compile(r"Given a board size of (\d+)x\d+,")
        self.board_size = int(pat_board_size.search(state_string).group(1))
        pat_word_list = re.compile(r"List of words:\n((- [a-z]+\n)+)\nPrint only the answer.")
        word_list_str = pat_word_list.search(state_string).group(1).strip()
        self.word_list = sorted(map(lambda t: t.strip("- "), word_list_str.split('\n')))

    def _get_prompt(self) -> str:
        prompt = (
            f"Given a board size of {self.board_size}x{self.board_size}, "
            "arrange a possible crossword puzzle answer from a list of words.\n"
            "Item in the list can only be used once.\n"
        )

        prompt += "\nList of words:\n"
        for word in self.word_list:
            prompt += f"- {word}\n"

        prompt += "\nPrint only the answer."
        return prompt

    def _validate(self, answer: str) -> (bool, str):
        answer = answer if answer else ""
        # ans_hor = list(filter(None, answer.lower().replace(' ', '\n').split("\n")))
        ans_hor = answer.lower().split()
        val_msg = ""
        if len(ans_hor) != self.board_size:
            arr = answer.lower().split()
            if all(len(l) == 1 for l in arr) and (len(arr) == self.board_size * self.board_size):
                ans_hor = ["".join(arr[i:i+self.board_size]) for i in range(0, len(arr), self.board_size)]
        if len(ans_hor) != self.board_size:
            val_msg = f"Mismatch answer length found!! Expected size of {self.board_size}, got {len(ans_hor)}."
            return False, val_msg
        for w in ans_hor:
            if len(w) != self.board_size:
                val_msg = f"Mismatch answer length found!! Expected size of {self.board_size}, got {len(w)}."
                return False, val_msg
        ans_ver = [''.join(ans_hor[r][c] for r in range(self.board_size)) for c in range(self.board_size)]
        word_set = set(self.word_list)
        for i, w in enumerate(chain(ans_hor, ans_ver)):
            if w not in word_set:
                val_msg = (f"Mismatch answer word found!! {'Horizontal' if i < self.board_size else 'Vertical'} word"
                           f" '{w}' is not in the word set.")
                return False, val_msg
            word_set.remove(w)
        return True, val_msg

    @staticmethod
    def example() -> (str, str):
        prompt = (f"Given a board size of 3x3, arrange a possible crossword puzzle answer from a list of words.\n"
                  f"Item in the list can only be used once.\n\n"
                  f"List of words:\n"
                  f"- app\n"
                  f"- all\n"
                  f"- and\n"
                  f"- lee\n"
                  f"- let\n"
                  f"- pat\n"
                  f"- pee\n"
                  f"- pet\n\n"
                  f"Print only the answer.")
        answer = "app\nlee\nlet"
        return prompt, answer

#%%


#%%


#%%



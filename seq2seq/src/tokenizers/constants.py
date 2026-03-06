base_token_list = [
    " ",
    "!",
    "#",
    "$",
    "%",
    "&",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "=",
    ">",
    "@",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "H",
    "I",
    "N",
    "O",
    "P",
    "R",
    "S",
    "U",
    "X",
    "[",
    "]",
    "c",
    "i",
    "l",
    "n",
    "o",
    "r",
    "s",
    "t",
    "{",
    "}",
]

special_token_list = ["[pad]", "[bos]", "[eos]", "[msk]", "[unk]"]

# the regular expression to handle patterns for tokens

import re

# special tokens are added as initial to ensure excape from singular breakdown precedence
all_tokens = special_token_list + base_token_list
escaped_tokens = [re.escape(token) for token in all_tokens]
token_pattern = "|".join(escaped_tokens) + "|."

# test_strin = "[eos]cco2_35[bos]"
# print(re.findall(token_pattern, test_strin))

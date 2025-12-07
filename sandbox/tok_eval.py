import itertools
import json
import os
import random

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer

benchmark_tokenizer_dir_name = os.getenv("BENCHMARK_TOKENIZER_DIR_NAME", "benchmark_tokenizer")
tokenizer = get_tokenizer()

test_words = [
    "authoritatively",
    "ballistic",
    "calisthenic",
    "caroler",
    "cedrorun",
    "collapsible",
    "contentiousness",
    "devilish",
    "dideoxyinosine",
    "dilettanteish",
    "disorient",
    "formularize",
    "formulating",
    "hemidemisemiquaver",
    "hypermastigote",
    "impracticable",
    "infotainment",
    "nonbeliever",
    "raucous",
    "spreader",
    "theoretically",
    "unsubstantialise",
]


def tokenized_for_comparison(text: str, benchmark=None):
    enc = tokenizer.encode(text)
    return (
        None if not benchmark else [benchmark.decode([tok]) for tok in benchmark.encode(text)],
        [tokenizer.decode([tok]) for tok in enc]
    )


if __name__ == '__main__':
    # Words in tok_eval.json are dumped from NLTK's WordNet (oewn2024), including all words that tokenize
    # into more than a single token.
    with open(f'{os.path.dirname(os.path.abspath(__file__))}/tok_eval.json') as f:
        eval_words = json.load(f)

    benchmark_tokenizer = (
        get_tokenizer(benchmark_tokenizer_dir_name)
        if os.path.isdir(os.path.join(get_base_dir(), benchmark_tokenizer_dir_name))
        else None
    )

    seed = os.getenv("RANDOM_SEED")
    if seed:
        random.seed(seed)
    static_list = list(itertools.chain.from_iterable([[word.capitalize(), word] for word in test_words]))
    for word in ([f" {word}" for word in static_list] + random.sample(eval_words, 10)):
        benchmark_tokens, tokens = tokenized_for_comparison(word, benchmark=benchmark_tokenizer)
        print("---")
        print(f"word:      {word.lstrip()}")
        if benchmark_tokens:
            print(f"benchmark: {benchmark_tokens}")
        print(f"tokens:    {tokens}")

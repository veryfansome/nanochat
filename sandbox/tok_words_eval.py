import os

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer

benchmark_tokenizer_dir_name = os.getenv("BENCHMARK_TOKENIZER_DIR_NAME", "benchmark_tokenizer")
tokenizer = get_tokenizer()

test_words = (
    "affixes",
    "anemia",
    "antidepressant",
    "antidisestablishmentarianism",
    "beneficiaries",
    "biology",
    "breakfast",
    "candidate",
    "closure",
    "collaborating",
    "corruption",
    "cupboard",
    "definite",
    "definition",
    "disproportionately",
    "erupted",
    "eruption",
    "execution",
    "exposure",
    "failure",
    "hematology",
    "hematoma",
    "hemorrhage",
    "hogwash",
    "hydrophobic",
    "hypoglycemia",
    "infinitesimal",
    "institution",
    "interruption",
    "leukemia",
    "librarian",
    "measure",
    "microscopic",
    "misunderstood",
    "mitochondrial",
    "mixture",
    "monomorphemic",
    "nation",
    "national",
    "natural",
    "nature",
    "people",
    "person",
    "pressure",
    "recognition",
    "recognize",
    "revolution",
    "running",
    "scope",
    "septicemia",
    "solution",
    "something",
    "structure",
    "toxemia",
    "uncharacteristically",
    "understand",
    "understood",
)


def tokenized_for_comparison(text: str, benchmark=None):
    enc = tokenizer.encode(text)
    return (
        None if not benchmark else [benchmark.decode([tok]) for tok in benchmark.encode(text)],
        [tokenizer.decode([tok]) for tok in enc]
    )


if __name__ == '__main__':
    benchmark_tokenizer = (
        get_tokenizer(benchmark_tokenizer_dir_name)
        if os.path.isdir(os.path.join(get_base_dir(), benchmark_tokenizer_dir_name))
        else None
    )
    for word in test_words:
        for test in [word.capitalize(), word]:
            benchmark_tokens, tokens = tokenized_for_comparison(f" {test}", benchmark=benchmark_tokenizer)
            print("---")
            print(f"word:      {test}")
            if benchmark_tokens:
                print(f"benchmark: {benchmark_tokens}")
            print(f"tokens:    {tokens}")

    #show_tokenized_text("Hello world!")

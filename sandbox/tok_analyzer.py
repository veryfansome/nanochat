from collections import Counter

from nanochat.dataset import parquets_iter_batched


def doc_iterator(max_docs: int = 1):
    """Similar to text_iterator() from scripts/tok_train.py"""
    ndocs = 0
    for batch in parquets_iter_batched(split="train"):
        for _doc in batch:
            if ndocs == max_docs:
                return
            yield _doc
            ndocs += 1


if __name__ == "__main__":
    from nanochat.tokenizer import get_tokenizer

    #tokenizer = get_tokenizer(tokenizer_dir_name="benchmark_tokenizer")
    tokenizer = get_tokenizer(tokenizer_dir_name="tokenizer")

    bigram_counts = Counter()
    for doc in doc_iterator(max_docs=10000):
        enc = tokenizer.encode(doc)
        tokens = [tokenizer.decode([tok]) for tok in enc]
        bigrams = zip(tokens, tokens[1:])
        bigram_counts.update(bigrams)

    for bigram, count in bigram_counts.most_common(n=100):
        print(bigram, count)

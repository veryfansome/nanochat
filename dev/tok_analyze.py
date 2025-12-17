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
    import argparse
    import os
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer

    base_dir = get_base_dir()

    parser = argparse.ArgumentParser(description='Token analyzer')
    parser.add_argument('--bigrams', type=bool, default=False, help='Analyze bigrams (default: False)')
    parser.add_argument('--max_docs', type=int, default=10_000,
                        help='Maximum documents to analyze (default: 10k)')
    parser.add_argument('--tokenizer_dir', default="tokenizer",
                        help=f'Tokenizer directory name under {base_dir} (default: tokenizer)')
    parser.add_argument('--top', type=int, default=100,
                        help='Number of top results to show (default: 100)')
    args = parser.parse_args()
    print(f"bigrams: {args.bigrams}")
    print(f"max_docs: {args.max_docs:,}")
    print(f"tokenizer_dir: {args.tokenizer_dir}")
    print(f"top: {args.top:,}")

    tokenizer = RustBPETokenizer.from_directory(os.path.join(base_dir, args.tokenizer_dir))

    if args.bigrams:
        bigram_counter = Counter()
        for doc in doc_iterator(max_docs=args.max_docs):
            enc = tokenizer.encode(doc)
            tokens = [tokenizer.decode([tok]) for tok in enc]
            bigrams = zip(tokens, tokens[1:])
            bigram_counter.update(bigrams)
        for bigram, count in bigram_counter.most_common(n=args.top):
            print(bigram, count)
    else:
        token_counter = Counter()
        for doc in doc_iterator(max_docs=args.max_docs):
            enc = tokenizer.encode(doc)
            tokens = [tokenizer.decode([tok]) for tok in enc]
            token_counter.update(tokens)
        for token, count in token_counter.most_common(n=args.top):
            print((token,), count)

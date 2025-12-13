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

    #tokenizer = get_tokenizer(tokenizer_dir_name="benchmark_tokenizer")  # 1st pass
    # (' of', ' the') 59985
    # (',', ' and') 39328
    # (' in', ' the') 36504
    # ('.', ' The') 26706
    # (',', ' the') 24389
    # (' to', ' the') 21248
    # (' ', '20') 17433
    # (' ', '19') 15716
    # (',', ' ') 14203
    # (' on', ' the') 14128
    # (' and', ' the') 14119
    # ('.\n', 'The') 10654
    # ('.\n', '-') 10080
    # (',', ' but') 10044
    # (' to', ' be') 10043
    # (' is', ' a') 9848
    # (' for', ' the') 9699
    # (' from', ' the') 9560
    # ('.', ' In') 9383
    # ('\n', '-') 9218
    # ('.', ' It') 8858
    # ('.', ' This') 8855
    # (' of', ' a') 8551
    # (' with', ' the') 8514
    # (',', ' which') 7983
    # ('.', ' ') 7763
    # (' in', ' a') 7715
    # (' by', ' the') 7590
    # (' can', ' be') 7524
    # (' at', ' the') 7384
    # (' in', ' ') 7356
    # (',', ' a') 7183
    # (' is', ' the') 7136
    # (' that', ' the') 6976
    # (' ', '1') 6918
    # (' as', ' a') 6868
    # (',', ' it') 6844
    # (',', ' or') 6289
    # (' it', ' is') 6272
    # ('�', '�') 6004
    # (' such', ' as') 5980
    # (' ', '2') 5633
    # (' ', '10') 5519
    # (' ', '18') 5387
    # (' the', ' ') 5153
    # (',', ' as') 4992
    # (' as', ' the') 4970
    # (' with', ' a') 4838
    # (',', ' in') 4725
    # (' to', ' a') 4707
    # (' ', '3') 4471
    # (' of', ' ') 4375
    # (' the', ' same') 4374
    # ('.', ' A') 4272
    # ('.', ' They') 4086
    # (' to', ' ') 4072
    # (' have', ' been') 3976
    # (' one', ' of') 3961
    # (' will', ' be') 3859
    # (' has', ' been') 3821
    # (' the', ' first') 3803
    # ('.', ' If') 3787
    # (' for', ' a') 3721
    # (',', ' you') 3683
    # (' is', ' not') 3667
    # (' the', ' most') 3634
    # (',', ' we') 3604
    # ('.', ' (') 3563
    # ('.\n', 'In') 3561
    # (' the', ' world') 3539
    # (' and', ' ') 3475
    # (' may', ' be') 3455
    # (',', ' they') 3428
    # (' as', ' well') 3402
    # ('.', ' For') 3353
    # (' into', ' the') 3304
    # ('00', '0') 3288
    # ('.', ' But') 3279
    # (' It', ' is') 3255
    # ('.', ' He') 3248
    # (' and', ' a') 3223
    # (',', ' with') 3218
    # (' part', ' of') 3181
    # (' ', '4') 3129
    # (',', '00') 3113
    # (' ', '5') 3077
    # (' of', ' this') 3073
    # ('\n', 'The') 3035
    # (' on', ' a') 3025
    # (' number', ' of') 2935
    # (' they', ' are') 2920
    # (' have', ' a') 2896
    # (' you', ' can') 2889
    # (' more', ' than') 2881
    # (' need', ' to') 2871
    # ('.', ' We') 2847
    # (' ', '15') 2773
    # (' about', ' the') 2773
    # ('.', ' These') 2771
    # (' However', ',') 2711

    tokenizer = get_tokenizer(tokenizer_dir_name="tokenizer")  # Additional passes for 3 and 4-grams?

    bigram_counts = Counter()
    for doc in doc_iterator(max_docs=10000):
        enc = tokenizer.encode(doc)
        tokens = [tokenizer.decode([tok]) for tok in enc]
        bigrams = zip(tokens, tokens[1:])
        bigram_counts.update(bigrams)

    for bigram, count in bigram_counts.most_common(n=100):
        print(bigram, count)

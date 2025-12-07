import rustbpe
import tiktoken
import pytest


def build_enc(tok: rustbpe.Tokenizer):
    """Build a tiktoken.Encoding from a trained rustbpe tokenizer."""
    pattern = tok.get_pattern()
    mergeable_ranks_list = tok.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )
    return enc, mergeable_ranks

# -----------------------------------------------------------------------------
# Seed token (warm-start merges) tests

def test_seed_token_cannot_cross_regex_chunk_boundaries():
    # Even if we seed "a\nb", encoding splits into chunks ("a", "\n", "b") so it
    # cannot become one token in this tokenizer design.
    text = "aaaa bbbb"
    vocab_size = 256 + 64

    tok = rustbpe.Tokenizer()
    tok.train_from_iterator([text], vocab_size, seed_tokens=["a\nb"])

    ids = tok.encode("a\nb")
    assert len(ids) > 1, "Seed token spanning regex chunks should not encode as a single token"

def test_seed_tokens_empty_list_is_noop():
    text = "the quick brown fox jumps over the lazy dog " * 5
    vocab_size = 256 + 20

    tok_none = rustbpe.Tokenizer()
    tok_none.train_from_iterator([text], vocab_size)

    tok_empty = rustbpe.Tokenizer()
    tok_empty.train_from_iterator([text], vocab_size, seed_tokens=[])

    # Strong check: identical training artifact
    assert tok_none.get_mergeable_ranks() == tok_empty.get_mergeable_ranks()

    # And encoding should match
    sample = "the quick brown fox"
    assert tok_none.encode(sample) == tok_empty.encode(sample)

def test_seed_tokens_merge_budget_too_small_raises():
    # "hello" needs 4 merges, but vocab only allows 3
    text = "aaaa bbbb"
    vocab_size = 256 + 3

    tok = rustbpe.Tokenizer()
    with pytest.raises(ValueError, match=r"Seed tokens require at most 4 merges"):
        tok.train_from_iterator([text], vocab_size, seed_tokens=["hello"])

def test_seed_tokens_unicode_creates_single_token():
    # Multi-byte UTF-8 seed token
    seed = "你好"  # 6 bytes in UTF-8
    text = "aaaa bbbb cccc"  # does not contain seed
    vocab_size = 256 + 64

    tok = rustbpe.Tokenizer()
    tok.train_from_iterator([text], vocab_size, seed_tokens=[seed])

    enc, ranks = build_enc(tok)

    seed_bytes = seed.encode("utf-8")
    assert seed_bytes in ranks, "Seed token bytes must appear in mergeable_ranks"

    ids_rust = tok.encode(seed)
    ids_tk = enc.encode(seed)
    assert ids_rust == ids_tk
    assert len(ids_rust) == 1, "Unicode seed should encode as a single token id"
    assert enc.decode(ids_rust) == seed

def test_seed_tokens_warm_start_creates_expected_token_and_ids():
    # Train on text that does NOT contain "hello" so the only way it becomes
    # a single token is via seed merges.
    text = "aaaa aaaa aaaa bbbb bbbb cccc cccc"
    vocab_size = 256 + 20

    tok = rustbpe.Tokenizer()
    tok.train_from_iterator([text], vocab_size, seed_tokens=["hello"])

    enc, ranks = build_enc(tok)

    # Seed chain for "hello" (bytes: h e l l o) should allocate:
    # (h,e)->256 => b"he"
    # (he,l)->257 => b"hel"
    # (hel,l)->258 => b"hell"
    # (hell,o)->259 => b"hello"
    assert ranks[b"he"] == 256
    assert ranks[b"hel"] == 257
    assert ranks[b"hell"] == 258
    assert ranks[b"hello"] == 259

    # Rust encode should produce the seed token as a single id.
    assert tok.encode("hello") == [259]

    # Exported tiktoken encoding must match rustbpe encode.
    assert enc.encode("hello") == [259]
    assert enc.decode([259]) == "hello"

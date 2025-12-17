import pytest

from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()


def _tok_assert(text, expected_tokens):
    enc = tokenizer.encode(text)
    tokens = [tokenizer.decode([tok]) for tok in enc]
    assert tokens == expected_tokens, f"Expected {expected_tokens}, got {tokens}"

@pytest.mark.parametrize("text, expected", [
    (" I am", [" I", " am"]),
])
def test_of_the(text: str, expected: list[str]):
    _tok_assert(text, expected)

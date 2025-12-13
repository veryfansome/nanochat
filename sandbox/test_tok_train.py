import pytest

from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()


def _test_tokenization(text, expected_tokens):
    enc = tokenizer.encode(text)
    tokens = [tokenizer.decode([tok]) for tok in enc]
    assert tokens == expected_tokens, f"Expected {expected_tokens}, got {tokens}"


@pytest.mark.parametrize("text, expected_tokens", [
    ("To be, or not to be", [
        "To", " be", ", or", " not", " to be"]),
    ("He did forget to get us the milk and the eggs in the end", [
        "He", " did", " forget", " to", " get", " us", " the", " milk", " and the", " eggs", " in the", " end"]),
    ("Please move to the left", [
        "Please", " move", " to the", " left"]),
    ("I saw him on the train", [
        "I", " saw", " him", " on the", " train"]),
    ("I saw him for the last time at the party", [
        "I", " saw", " him", " for the", " last", " time", " at the", " party"]),
    ("He is in a real bind", [
       "He", " is", " in a", " real", " bind"]),
    ("He is a teacher, and he is the principal", [
        "He", " is a", " teacher", ", and", " he", " is the", " principal"]),
    ("I'm of the opinion that he is a son of a gun", [
        "I", "'m", " of the", " opinion", " that", " he", " is a", " son", " of a", " gun"]),
    ("He is on a roll", [
        "He", " is", " on a", " roll", ]),
    ("He went to a party", [
        "He", " went", " to a", " party", ]),
    ("He is from the docks", [
       "He", " is", " from the", " docks", ]),
    ("He is skilled with the knife", [
       "He", " is", " skilled", " with the", " knife", ]),
    ("He does things by the book", [
       "He", " does", " things", " by the", " book"]),
    ("He knows about the crimes", [
        "He", " knows", " about the", " crimes", ]),
    ("He went into the bush", [
        "He", " went", " into the", " bush"]),
    ("Is that the room?", [
       "Is", " that the", " room", "?"]),
    ("I went as a clown", [
       "I", " went", " as a", " clown"]),
    ("He retired as the head coach", [
       "He", " retired", " as the", " head", " coach"]),
    ("I will be back", [
       "I", " will be", " back"]),
    ("There is more than enough", [
       "There", " is", " more than", " enough"]),
    ("It can be", [
        "It", " can be"]),

    ("He went to work, and she went out", [
        "He", " went", " to", " work", ", and", " she", " went", " out",]),
    ("When we finished our work, the rain stopped", [
        "When", " we", " finished", " our", " work", ", the", " rain", " stopped"]),
    ("I like him, but he drinks", [
       "I", " like", " him", ", but", " he", " drinks"]),
    ("You can wait, which costs nothing", [
        "You", " can", " wait", ", which", " costs", " nothing" ]),
    ("We ate the pig, and, amazingly, it tasted great", [
        "We", " ate", " the", " pig", ", and", ",", " amazingly", ", it", " tasted", " great"]),
])
def test_forced_merges(text: str, expected_tokens: list[str]):
    _test_tokenization(text, expected_tokens)

@pytest.mark.parametrize("text, expected", [
    ("You can eat the pig, or not", ["You", " can", " eat", " the", " pig", ", or", " not"]),
])
def test_comma_or(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("I think it is nice", ["I", " think", " it is", " nice"]),
    ("I think it is the dog", ["I", " think", " it is", " the", " dog"]),
])
def test_it_is(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("You can find produce at the market, such as apples", ["You", " can", " find", " produce", " at the", " market", ",", " such as", " apples"]),
])
def test_such_as(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("I have to tell you, as your teacher", ["I", " have", " to", " tell", " you", ", as", " your", " teacher"]),
])
def test_comma_as(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("We saw a deer, in the backyard", ["We", " saw", " a", " deer", ", in", " the", " backyard"]),
])
def test_comma_in(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He came back with a gun", ["He", " came", " back", " with a", " gun"]),
])
def test_with_a(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("I have been a teacher", ["I", " have been", " a", " teacher"]),
])
def test_have_been(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("I am but one of many", ["I", " am", " but", " one of", " many"]),
])
def test_one_of(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He has been a teacher", ["He", " has been", " a", " teacher"]),
])
def test_has_been(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
        ("He looked tired, you know", ["He", " looked", " tired", ", you", " know"]),
])
def test_comma_you(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
        ("Don't say it, we know", ["Don", "'t", " say", " it", ", we", " know"]),
])
def test_comma_we(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
        ("It may be true", ["It", " may be", " true"]),
])
def test_may_be(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("They know as well", ["They", " know", " as well"]),
])
def test_as_well(text: str, expected: list[str]):
    _test_tokenization(text, expected)

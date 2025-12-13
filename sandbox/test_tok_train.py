import pytest

from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()


def _test_tokenization(text, expected_tokens):
    enc = tokenizer.encode(text)
    tokens = [tokenizer.decode([tok]) for tok in enc]
    assert tokens == expected_tokens, f"Expected {expected_tokens}, got {tokens}"

@pytest.mark.parametrize("text, expected", [
    ("I'm of the opinion",
     ["I", "'m", " of the", " opinion"]),
])
def test_of_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He went to work, and she went out",
     ["He", " went", " to", " work", ", and", " she", " went", " out"]),
])
def test_comma_and(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He did forget in the end",
     ["He", " did", " forget", " in the", " end"]),
])
def test_in_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He did forget. The goal was dropped",
     ["He", " did", " forget", ". The", " goal", " was", " dropped"]),
])
def test_period_The(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("When we finished our work, the rain stopped",
     ["When", " we", " finished", " our", " work", ", the", " rain", " stopped"]),
])
def test_comma_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("Please move to the left",
     ["Please", " move", " to the", " left"]),
])
def test_to_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

#@pytest.mark.parametrize("text, expected", [
#    ("In the years 2020, 2022, and 2023",
#     ["In", " the", " years", " ", "20", "20", ", ", "20", "22", ", and", " ", "20", "23"]),
#])
#def test_comma_space(text: str, expected: list[str]):
#    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("I saw him on the train",
     ["I", " saw", " him", " on the", " train"]),
])
def test_on_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

#@pytest.mark.parametrize("text, expected", [
#    ("Get us the milk and the eggs",
#     ["Get", " us", " the", " milk", " and the", " eggs"]),
#])
#def test_and_the(text: str, expected: list[str]):
#    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("To be, or not to be",
     ["To", " be", ", or", " not", " to be"]),
])
def test_to_be(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("I like him, but he drinks",
     ["I", " like", " him", ", but", " he", " drinks"]),
])
def test_comma_but(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He is a teacher",
     ["He", " is a", " teacher"]),
    ("He is another teacher",
     ["He", " is", " another", " teacher"]),
])
def test_is_a(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("I saw him for the last time",
     ["I", " saw", " him", " for the", " last", " time"]),
])
def test_for_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He is from the docks",
     ["He", " is", " from the", " docks"]),
])
def test_from_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("You should wait. In June, it gets warm again",
     ["You", " should", " wait", ". In", " June", ", it", " gets", " warm", " again"]),
])
def test_period_In(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("You should wait. It gets warm in a month",
     ["You", " should", " wait", ". It", " gets", " warm", " in a", " month"]),
])
def test_period_It(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("We lost. This sucks",
     ["We", " lost", ". This", " sucks"]),
])
def test_period_This(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He is a son of a gun",
     ["He", " is a", " son", " of a", " gun"]),
])
def test_of_a(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He is skilled with the knife",
     ["He", " is", " skilled", " with the", " knife"]),
])
def test_with_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("You can wait, which costs nothing",
     ["You", " can", " wait", ", which", " costs", " nothing"]),
])
def test_comma_which(text: str, expected: list[str]):
    _test_tokenization(text, expected)

#@pytest.mark.parametrize("text, expected", [
#    (" meant war. 1066 would mark",
#     [" meant", " war", ". ", "10", "66", " would", " mark"]),
#])
#def test_period_space(text: str, expected: list[str]):
#    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He is in a real bind",
     ["He", " is", " in a", " real", " bind"]),
])
def test_in_a(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He does things by the book",
     ["He", " does", " things", " by the", " book"]),
])
def test_by_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("It can be",
     ["It", " can be"]),
])
def test_can_be(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("I saw him at the party",
     ["I", " saw", " him", " at the", " party"]),
])
def test_at_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

#@pytest.mark.parametrize("text, expected", [
#    ("He died in 1346",
#     ["He", " died", " in ", "13", "46"]),
#])
#def test_in_space(text: str, expected: list[str]):
#    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He gave her a rose, an expression of his love",
     ["He", " gave", " her", " a", " rose", ",", " an", " expression", " of", " his", " love"]),
    ("I saw a sports car, a shiny red one",
     ["I", " saw", " a", " sports", " car", ", a", " shiny", " red", " one"]),
    ("I saw another sports car, another shiny red one",
     ["I", " saw", " another", " sports", " car", ",", " another", " shiny", " red", " one"]),
])
def test_comma_a(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He is the principal",
     ["He", " is the", " principal"]),
])
def test_is_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("Is that the room?",
     ["Is", " that the", " room", "?"]),
])
def test_that_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("I went as a clown",
     ["I", " went", " as a", " clown"]),
])
def test_as_a(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("We ate the pig, and, amazingly, it tasted great",
     ["We", " ate", " the", " pig", ", and", ",", " amazingly", ", it", " tasted", " great"]),
])
def test_comma_it(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("You can eat the pig, or not",
     ["You", " can", " eat", " the", " pig", ", or", " not"]),
])
def test_comma_or(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("I think it is nice",
     ["I", " think", " it is", " nice"]),
    ("I think it is the dog",
     ["I", " think", " it is", " the", " dog"]),
])
def test_it_is(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("You can find produce at the market, such as apples",
     ["You", " can", " find", " produce", " at the", " market", ",", " such as", " apples"]),
])
def test_such_as(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("Many people died in the 1340s",
     ["Many", " people", " died", " in", " the ", "13", "40", "s"]),
])
def test_comma_as(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("I have to tell you, as your teacher",
     ["I", " have", " to", " tell", " you", ", as", " your", " teacher"]),
])
def test_comma_as(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He retired as the head coach",
     ["He", " retired", " as the", " head", " coach"]),
])
def test_as_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He came back with a gun",
     ["He", " came", " back", " with a", " gun"]),
])
def test_with_a(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("We lost. They won",
     ["We", " lost", ". They", " won"]),
])
def test_period_They(text: str, expected: list[str]):
    _test_tokenization(text, expected)

#@pytest.mark.parametrize("text, expected", [
#    ("It's just not the same",
#     ["It", "'s", " just", " not", " the same"]),
#    ("It is the same",
#     ["It", " is the", " same"]),
#    ("Yes, and the same thing happened",
#     ["Yes", ", and", " the same", " thing", " happened"]),
#])
#def test_the_same(text: str, expected: list[str]):
#    _test_tokenization(text, expected)

#@pytest.mark.parametrize("text, expected", [
#    # TODO: this breaks up in-the
#    ("We saw a deer, in the backyard",
#     ["We", " saw", " a", " deer", ", in", " the", " backyard"]),
#])
#def test_comma_in(text: str, expected: list[str]):
#    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He went to a party",
     ["He", " went", " to a", " party"]),
])
def test_to_a(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("I have been a teacher",
     ["I", " have been", " a", " teacher"]),
])
def test_have_been(text: str, expected: list[str]):
    _test_tokenization(text, expected)

#@pytest.mark.parametrize("text, expected", [
#    ("I am but one of many",
#     ["I", " am", " but", " one of", " many"]),
#    ("I am one of the students",
#     ["I", " am", " one of", " the", " students"]),  # Rather than " one", " of the"
#])
#def test_one_of(text: str, expected: list[str]):
#    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("I will be back",
     ["I", " will be", " back"]),
])
def test_will_be(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He has been a teacher",
     ["He", " has been", " a", " teacher"]),
])
def test_has_been(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("We lost. If we only practiced more",
     ["We", " lost", ". If", " we", " only", " practiced", " more"]),
])
def test_period_If(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He is fast for a slug",
     ["He", " is", " fast", " for a", " slug"]),
])
def test_for_a(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He looked tired, you know",
     ["He", " looked", " tired", ", you", " know"]),
])
def test_comma_you(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("Don't say it, we know",
     ["Don", "'t", " say", " it", ", we", " know"]),
])
def test_comma_we(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("It may be true",
     ["It", " may be", " true"]),
])
def test_may_be(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("Don't help them, they don't need it",
     ["Don", "'t", " help", " them", ", they", " don", "'t", " need", " it"]),
])
def test_comma_they(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("They know as well",
     ["They", " know", " as well"]),
])
def test_as_well(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He spoke up. For his country, he would risk it all",
     ["He", " spoke", " up", ". For", " his", " country", ",", " he", " would", " risk", " it", " all"]),
])
def test_period_For(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He went into the bush",
     ["He", " went", " into the", " bush"]),
])
def test_into_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He lost. But he put up a good fight",
     ["He", " lost", ". But", " he", " put", " up", " a", " good", " fight"]),
])
def test_period_But(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He will lose. It is guaranteed",
     ["He", " will", " lose", ". It", " is", " guaranteed"]),
])
def test_It_is(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He came. He saw",
     ["He", " came", ". He", " saw"]),
])
def test_period_He(text: str, expected: list[str]):
    _test_tokenization(text, expected)

#@pytest.mark.parametrize("text, expected", [
#    ("He wanted a dog and a cat",
#     ["He", " wanted", " a", " dog", " and a", " cat"]),
#    ("He wanted a bird, a dog, and a cat",
#     ["He", " wanted", " a", " bird", ", a", " dog", ", and", " a", " cat"]),  # Rather than ",", " and a"
#])
#def test_and_a(text: str, expected: list[str]):
#    _test_tokenization(text, expected)

#@pytest.mark.parametrize("text, expected", [
#    ("Order a pizza, with extra cheese",
#     ["Order", " a", " pizza", ", with", " extra", " cheese"]),
#    ("Order a pizza, without meat",
#     ["Order", " a", " pizza", ",", " without", " meat"]),
#])
#def test_comma_with(text: str, expected: list[str]):
#    _test_tokenization(text, expected)

#@pytest.mark.parametrize("text, expected", [
#    ("He is part of our class",
#     ["He", " is", " part of", " our", " class"]),
#    ("He is part of the crew",
#     ["He", " is", " part of", " the", " crew"]),  # Rather than " part", " of the"
#])
#def test_part_of(text: str, expected: list[str]):
#    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He is tired of this game",
     ["He", " is", " tired", " of this", " game"]),
])
def test_of_this(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He is on a roll",
     ["He", " is", " on a", " roll"]),
])
def test_on_a(text: str, expected: list[str]):
    _test_tokenization(text, expected)

#@pytest.mark.parametrize("text, expected", [
#    ("There were a number of mistakes",
#     ["There", " were", " a", " number of", " mistakes"]),
#    ("A number of the classes are free",
#     ["A", " number of", " the", " classes", " are", " free"]),  # Rather than " number", " of the"
#])
#def test_number_of(text: str, expected: list[str]):
#    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("Please, have a treat",
     ["Please", ",", " have a", " treat"]),
])
def test_have_a(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("There is more than enough",
     ["There", " is", " more than", " enough"]),
])
def test_more_than(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("We lost. We played our best",
     ["We", " lost", ". We", " played", " our", " best"]),
])
def test_period_We(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He knows about the crimes",
     ["He", " knows", " about the", " crimes"]),
])
def test_about_the(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He lost. These guys are too good",
     ["He", " lost", ". These", " guys", " are", " too", " good"]),
])
def test_period_These(text: str, expected: list[str]):
    _test_tokenization(text, expected)

@pytest.mark.parametrize("text, expected", [
    ("He lost. However, he put up a good fight",
     ["He", " lost", ".", " However,", " he", " put", " up", " a", " good", " fight"]),
])
def test_However_comma(text: str, expected: list[str]):
    _test_tokenization(text, expected)

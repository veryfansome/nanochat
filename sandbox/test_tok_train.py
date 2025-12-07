import pytest

from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()


@pytest.mark.parametrize("word, expected_tokens", [
    (" Affixes", [' Af', 'fix', 'es']),
    (" affixes", [' affix', 'es']),

    (" Androgenous", [' Andr', 'ogenous']),
    (" androgenous", [' andr', 'ogenous']),

    (" Antidepressant", [' Anti', 'depress', 'ant']),
    (" antidepressant", [' antidepressant']),

    (" Antidisestablishmentarianism", [' Anti', 'dis', 'establish', 'ment', 'arian', 'ism']),
    (" antidisestablishmentarianism", [' anti', 'dis', 'establish', 'ment', 'arian', 'ism']),

    (" Appoint", [' Ap', 'point']),
    (" appoint", [' appoint']),

    (" Beneficiaries", [' Bene', 'fic', 'iaries']),
    (" beneficiaries", [' beneficiaries']),

    (" Closure", [' Clos', 'ure']),
    (" closure", [' closure']),

    (" Collaborating", [' Collabor', 'ating']),
    (" collaborating", [' collaborating']),

    (" Cupboard", [' Cup', 'board']),
    (" cupboard", [' cupboard']),

    (" Definite", [' De', 'finite']),
    (" definite", [' definite']),

    (" Disproportionately", [' Dis', 'pro', 'portion', 'ately']),
    (" disproportionately", [' disproportionately']),

    (" Erupted", [' E', 'rupted']),
    (" erupted", [' erupted']),

    (" Eruption", [' E', 'ruption']),
    (" eruption", [' eruption']),

    (" Hematology", [' Hemat', 'ology']),
    (" hematology", [' hemat', 'ology']),

    (" Hematoma", [' Hemat', 'oma']),
    (" hematoma", [' hemat', 'oma']),

    (" Hemorrhage", [' Hemo', 'rrhage']),
    (" hemorrhage", [' hemorrhage']),

    (" Heterocycle", [' Hetero', 'cycle']),
    (" heterocycle", [' hetero', 'cycle']),

    (" Hogwash", [' Hog', 'wash']),
    (" hogwash", [' hog', 'wash']),

    (" Hydrophobic", [' Hydro', 'phobic']),
    (" hydrophobic", [' hydrophobic']),

    (" Hypoglycemia", [' Hypo', 'glycemia']),
    (" hypoglycemia", [' hypoglycemia']),

    (" Infinitesimal", [' In', 'fin', 'it', 'esimal']),
    (" infinitesimal", [' infinit', 'esimal']),

    (" Interruption", [' Inter', 'ruption']),
    (" interruption", [' interruption']),

    (" Intramural", [' Intra', 'mural']),
    (" intramural", [' intra', 'mural']),

    (" Microscopic", [' Micro', 'scopic']),
    (" microscopic", [' microscopic']),

    (" Misunderstood", [' Mis', 'under', 'stood']),
    (" misunderstood", [' misunderstood']),

    (" Mitochondrial", [' Mito', 'chondr', 'ial']),
    (" mitochondrial", [' mitochondrial']),

    (" Mixture", [' Mix', 'ture']),
    (" mixture", [' mixture']),

    (" Monomorphemic", [' Mono', 'morph', 'emic']),
    (" monomorphemic", [' mono', 'morph', 'emic']),

    (" Noctambulation", [' Noct', 'ambul', 'ation']),
    (" noctambulation", [' noct', 'ambul', 'ation']),

    (" Septicemia", [' Septic', 'emia']),
    (" septicemia", [' septic', 'emia']),

    (" Staphylococcus", [' Staphylococcus']),
    (" staphylococcus", [' staphylo', 'coccus']),

    (" Suburbanise", [' Sub', 'urban', 'ise']),
    (" suburbanise", [' suburban', 'ise']),

    (" Syncretistic", [' Syn', 'cret', 'istic']),
    (" syncretistic", [' syn', 'cret', 'istic']),

    (" Toxemia", [' Tox', 'emia']),
    (" toxemia", [' tox', 'emia']),

    (" Uncharacteristically", [' Un', 'character', 'istically']),
    (" uncharacteristically", [' un', 'character', 'istically']),

    (" Understood", [' Under', 'stood']),
    (" understood", [' understood']),

    (" Unmindfully", [' Un', 'mind', 'fully']),
    (" unmindfully", [' un', 'mind', 'fully']),

    (" Unsegregated", [' Un', 'se', 'gregated']),
    (" unsegregated", [' un', 'se', 'gregated']),

    (" Washingtonian", [' Washington', 'ian']),
])
def test_words(word: str, expected_tokens: list[str]):
    enc = tokenizer.encode(word)
    tokens = [tokenizer.decode([tok]) for tok in enc]
    assert tokens == expected_tokens, f"Expected {expected_tokens}, got {tokens}"

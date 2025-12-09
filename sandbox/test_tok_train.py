import pytest

from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()


@pytest.mark.parametrize("word, expected_tokens", [
    (" Able", [' Able']),
    (" able", [' able']),

    (" Ability", [' Ability']),
    (" ability", [' ability']),

    (" Act", [' Act']),
    (" act", [' act']),

    (" Action", [' Action']),
    (" action", [' action']),

    (" Activity", [' Activity']),
    (" activity", [' activity']),

    (" Affixes", [' Af', 'fixes']),
    (" affixes", [' af', 'fixes']),

    (" Affordability", [' Afford', 'ability']),
    (" affordability", [' affordability']),

    (" Ambulance", [' Ambul', 'ance']),
    (" ambulance", [' ambulance']),

    (" Amble", [' Amble']),
    (" amble", [' amble']),

    (" Anachronistic", [' Ana', 'chron', 'istic']),
    (" anachronistic", [' anachron', 'istic']),

    (" Androgenous", [' Andr', 'o', 'genous']),
    (" androgenous", [' andr', 'o', 'genous']),

    (" Android", [' Android']),
    (" android", [' android']),

    (" Antidepressant", [' Anti', 'depress', 'ant']),
    (" antidepressant", [' antidepressant']),

    (" Antidisestablishmentarianism", [' Anti', 'dis', 'establish', 'ment', 'arian', 'ism']),
    (" antidisestablishmentarianism", [' anti', 'dis', 'establish', 'ment', 'arian', 'ism']),

    (" Appoint", [' Ap', 'point']),
    (" appoint", [' appoint']),

    (" Beneficiaries", [' Bene', 'fic', 'iaries']),
    (" beneficiaries", [' beneficiaries']),

    (" Bicycle", [' Bicycle']),
    (" bicycle", [' bicycle']),

    (" Breakfast", [' Breakfast']),
    (" breakfast", [' breakfast']),

    (" Breakfix", [' Break', 'fix']),
    (" breakfix", [' break', 'fix']),

    (" Calculatingly", [' Calculating', 'ly']),
    (" calculatingly", [' calculating', 'ly']),

    (" Capable", [' Cap', 'able']),
    (" capable", [' capable']),

    (" Circumambulate", [' Circum', 'ambul', 'ate']),
    (" circumambulate", [' circum', 'ambul', 'ate']),

    (" Closure", [' Clos', 'ure']),
    (" closure", [' closure']),

    (" Cognizant", [' Cogn', 'iz', 'ant']),
    (" cognizant", [' cognizant']),

    (" Collaborating", [' Collabor', 'ating']),
    (" collaborating", [' collaborating']),

    (" Counterproductive", [' Counter', 'productive']),
    (" counterproductive", [' counterproductive']),

    (" Counterterror", [' Counter', 'terror']),
    (" counterterror", [' counter', 'terror']),

    (" Credentials", [' Cred', 'entials']),
    (" credentials", [' credentials']),

    (" Cupboard", [' Cup', 'board']),
    (" cupboard", [' cupboard']),

    (" Definite", [' De', 'finite']),
    (" definite", [' definite']),

    (" Demigod", [' Demi', 'god']),
    (" demigod", [' demi', 'god']),

    (" Dermatology", [' Dermatology']),
    (" dermatology", [' dermatology']),

    (" Dermatosis", [' Dermat', 'osis']),
    (" dermatosis", [' dermat', 'osis']),

    (" Dermatoglyphics", [' Dermat', 'oglyph', 'ics']),
    (" dermatoglyphics", [' dermat', 'oglyph', 'ics']),

    (" dictation", [' dictation']),
    (" dictation", [' dictation']),

    (" Disproportionately", [' Dis', 'pro', 'portion', 'ately']),
    (" disproportionately", [' disproportionately']),

    (" Erupted", [' E', 'rupted']),
    (" erupted", [' erupted']),

    (" Eruption", [' E', 'ruption']),
    (" eruption", [' eruption']),

    (" Excrete", [' Ex', 'crete']),
    (" excrete", [' excrete']),

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

    (" Interconnect", [' Inter', 'connect']),
    (" interconnect", [' interconnect']),

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

    (" Monometallic", [' Mono', 'metallic']),
    (" monometallic", [' mono', 'metallic']),

    (" Motorcycle", [' Motor', 'cycle']),
    (" motorcycle", [' motorcycle']),

    (" Overrun", [' Over', 'run']),
    (" overrun", [' overrun']),

    (" Preamble", [' Pre', 'amble']),
    (" preamble", [' preamble']),

    (" Pushup", [' Push', 'up']),
    (" pushup", [' push', 'up']),

    (" Quintessence", [' Quint', 'essence']),
    (" quintessence", [' quint', 'essence']),

    (" Quintet", [' Quint', 'et']),
    (" quintet", [' quint', 'et']),

    (" Somnambulate", [' Somn', 'ambul', 'ate']),
    (" somnambulate", [' somn', 'ambul', 'ate']),

    (" Staphylococcus", [' Staphylococcus']),
    (" staphylococcus", [' staphylo', 'coccus']),

    (" Suburbanise", [' Sub', 'urban', 'ise']),
    (" suburbanise", [' suburban', 'ise']),

    (" Syncretistic", [' Syn', 'cret', 'istic']),
    (" syncretistic", [' syn', 'cret', 'istic']),

    (" Taxidermy", [' Taxi', 'derm', 'y']),
    (" taxidermy", [' taxi', 'derm', 'y']),

    (" Telephonics", [' Tele', 'phon', 'ics']),
    (" telephonics", [' tele', 'phon', 'ics']),

    (" Telephony", [' Tele', 'phony']),
    (" telephony", [' telephony']),

    (" Toxemia", [' Tox', 'emia']),
    (" toxemia", [' tox', 'emia']),

    (" Tricycle", [' Tri', 'cycle']),
    (" tricycle", [' tri', 'cycle']),

    (" Unbranded", [' Un', 'brand', 'ed']),
    (" unbranded", [' un', 'brand', 'ed']),

    (" Uncharacteristically", [' Un', 'character', 'istically']),
    (" uncharacteristically", [' un', 'character', 'istically']),

    (" Understood", [' Under', 'stood']),
    (" understood", [' understood']),

    (" Underweight", [' Under', 'weight']),
    (" underweight", [' underweight']),

    (" Unicycle", [' Uni', 'cycle']),
    (" unicycle", [' uni', 'cycle']),

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

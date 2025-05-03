from functools import cache

from .bow_lm import \
    EnglishGpt2Small, EnglishGpt2Medium, EnglishGpt2Large, EnglishGpt2Xl, \
    EnglishPythia70M, EnglishPythia160M, EnglishPythia410M, \
    EnglishPythia14B, EnglishPythia28B, EnglishPythia69B, EnglishPythia120B, \
    GermanGpt2Small

MODELS = {
    "gpt2-small": EnglishGpt2Small,
    "gpt2-medium": EnglishGpt2Medium,
    "gpt2-large": EnglishGpt2Large,
    "gpt2-xl": EnglishGpt2Xl,
    "pythia-70m": EnglishPythia70M,
    "pythia-160m": EnglishPythia160M,
    "pythia-410m": EnglishPythia410M,
    "pythia-14b": EnglishPythia14B,
    "pythia-28b": EnglishPythia28B,
    "pythia-69b": EnglishPythia69B,
    "pythia-120b": EnglishPythia120B,
    "dbmdz/german-gpt2-faust": GermanGpt2Small
}

LANGUAGES = {
    "gpt2-small": {"en"},
    "gpt2-medium": {"en"},
    "gpt2-large": {"en"},
    "gpt2-xl": {"en"},
    "pythia-70m": {"en"},
    "pythia-160m": {"en"},
    "pythia-410m": {"en"},
    "pythia-14b": {"en"},
    "pythia-28b": {"en"},
    "pythia-69b": {"en"},
    "pythia-120b": {"en"},
    "dbmdz/german-gpt2-faust": {"de"},
}

@cache
def get_model(model_name):
    model_cls = MODELS[model_name]
    return model_cls()

def get_bow_symbol(model_name):
    return MODELS[model_name].bow_symbol

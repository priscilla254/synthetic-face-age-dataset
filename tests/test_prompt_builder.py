import random

from facedata.generation.seed_generator import NEGATIVE_PROMPT, PromptBuilder


def test_same_seed_gives_same_prompt():
    a = PromptBuilder(random.Random(1)).sample("Arabic", "female", "25")
    b = PromptBuilder(random.Random(1)).sample("Arabic", "female", "25")
    assert a.prompt == b.prompt


def test_arabic_women_always_wear_hijab():
    builder = PromptBuilder(random.Random(0))
    for _ in range(30):
        assert "wearing a hijab" in builder.sample("Arabic", "female", "25").accessories


def test_no_durag_or_turban_for_other_groups():
    builder = PromptBuilder(random.Random(0))
    for _ in range(100):
        acc = builder.sample("Chinese", "male", "25").accessories
        assert "wearing a durag" not in acc and "wearing an Indian-style turban" not in acc


def test_minors_never_get_adult_outfits():
    builder = PromptBuilder(random.Random(0))
    for _ in range(100):
        outfit = builder.sample("Brazilian", "male", "13").outfit
        assert outfit in {"a school uniform", "casual clothes", "a hoodie", "sportswear"}


def test_prompt_contains_age_ethnicity_and_negative_prompt_is_non_empty():
    attrs = PromptBuilder(random.Random(3)).sample("Filipino", "male", "17")
    assert "17-year-old" in attrs.prompt and "Filipino" in attrs.prompt
    assert "watermark" in NEGATIVE_PROMPT

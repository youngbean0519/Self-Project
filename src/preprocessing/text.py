import re

def normalize_repeated_characters(text, max_repeat=2):
    pattern = r"(.)\1{" + str(max_repeat) + r",}"
    return re.sub(
        pattern,
        lambda match: match.group(1) * max_repeat,
        text,
    )

def remove_selected_symbols(text):
    text = re.sub(
        r"[^\w\s가-힣!?~ㅋㅎㅠㅜ]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def preprocess_text(text, normalize_repeat=False, remove_symbols=False):
    processed = text

    if normalize_repeat:
        processed = normalize_repeated_characters(processed)
    
    if remove_symbols:
        processed = remove_selected_symbols(processed)
    
    return processed
import ast
import json

import pandas as pd


TARGET_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]


def safe_parse_list(cell):
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return [str(item) for item in cell]
    if isinstance(cell, str):
        try:
            parsed = json.loads(cell)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (ValueError, SyntaxError, TypeError):
        pass
    return [str(cell)]


def safe_join_list_as_text(items):
    if not items:
        return ""
    return " ".join(str(item) for item in items)


def build_model_input(df):
    frame = df.copy()
    frame["prompt_list"] = frame["prompt"].apply(safe_parse_list)
    frame["response_a_list"] = frame["response_a"].apply(safe_parse_list)
    frame["response_b_list"] = frame["response_b"].apply(safe_parse_list)
    frame["prompt_text"] = frame["prompt_list"].apply(safe_join_list_as_text)
    frame["response_a_text"] = frame["response_a_list"].apply(safe_join_list_as_text)
    frame["response_b_text"] = frame["response_b_list"].apply(safe_join_list_as_text)
    return (
        frame["prompt_text"]
        + " [A] "
        + frame["response_a_text"]
        + " [B] "
        + frame["response_b_text"]
    )


def make_prediction_frame(pipeline, label_encoder, df):
    probs = pipeline.predict_proba(build_model_input(df))
    probabilities = pd.DataFrame(
        probs,
        columns=[f"winner_{name}" for name in label_encoder.classes_],
        index=df.index,
    )
    return probabilities.reindex(columns=TARGET_COLUMNS)

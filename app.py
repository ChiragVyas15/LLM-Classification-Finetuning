from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from model_utils import TARGET_COLUMNS, make_prediction_frame


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict_single(pipeline, label_encoder, prompt, response_a, response_b):
    df = pd.DataFrame(
        [
            {
                "prompt": prompt,
                "response_a": response_a,
                "response_b": response_b,
            }
        ]
    )
    return make_prediction_frame(pipeline, label_encoder, df).iloc[0]


st.set_page_config(page_title="LLM Response Classifier", page_icon="⚖️", layout="wide")

st.title("LLM Response Classifier")

if not MODEL_PATH.exists():
    st.error("model.joblib was not found. Run `python train_model.py` first.")
    st.stop()

artifact = load_model()
pipeline = artifact["pipeline"]
label_encoder = artifact["label_encoder"]
validation_log_loss = artifact.get("validation_log_loss")

if validation_log_loss is not None:
    st.caption(f"Validation log loss from saved training run: {validation_log_loss:.4f}")

tab_predict, tab_csv = st.tabs(["Single prediction", "CSV scoring"])

with tab_predict:
    prompt = st.text_area("Prompt", height=120, placeholder="Paste the user prompt here")
    col_a, col_b = st.columns(2)
    with col_a:
        response_a = st.text_area("Response A", height=260, placeholder="Paste model A response")
    with col_b:
        response_b = st.text_area("Response B", height=260, placeholder="Paste model B response")

    if st.button("Predict winner", type="primary", use_container_width=True):
        if not prompt.strip() or not response_a.strip() or not response_b.strip():
            st.warning("Please fill in the prompt, response A, and response B.")
        else:
            probs = predict_single(pipeline, label_encoder, prompt, response_a, response_b)
            winner_label = probs.idxmax()
            winner_name = {
                "winner_model_a": "Model A",
                "winner_model_b": "Model B",
                "winner_tie": "Tie",
            }[winner_label]

            st.subheader(f"Predicted winner: {winner_name}")
            st.dataframe(
                probs.rename(
                    {
                        "winner_model_a": "Model A",
                        "winner_model_b": "Model B",
                        "winner_tie": "Tie",
                    }
                )
                .mul(100)
                .round(2)
                .rename("Probability (%)"),
                use_container_width=True,
            )
            st.bar_chart(probs.rename({"winner_model_a": "Model A", "winner_model_b": "Model B", "winner_tie": "Tie"}))

with tab_csv:
    uploaded_file = st.file_uploader("Upload a CSV with prompt, response_a, and response_b columns", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        required_columns = {"prompt", "response_a", "response_b"}
        missing_columns = required_columns.difference(data.columns)

        if missing_columns:
            st.error(f"Missing required columns: {', '.join(sorted(missing_columns))}")
        else:
            probabilities = make_prediction_frame(pipeline, label_encoder, data)
            scored = data.copy()
            for column in TARGET_COLUMNS:
                scored[column] = probabilities[column]

            if "id" in scored.columns:
                submission = scored[["id", *TARGET_COLUMNS]]
            else:
                submission = scored[TARGET_COLUMNS]

            st.dataframe(submission.head(50), use_container_width=True)
            st.download_button(
                "Download predictions",
                data=submission.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

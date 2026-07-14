# LLM-Classification-Finetuning
LLM Response Classifier using TF-IDF and Logistic Regression to predict whether Model A, Model B, or Tie provides the better response. Includes Streamlit UI, batch CSV prediction, and automated submission file generation.

# 🤖 LLM Response Classifier

A machine learning application that predicts whether **Model A**, **Model B**, or **Tie** provides the better response for a given prompt. The project uses **TF-IDF vectorization** and **Logistic Regression** to classify responses and provides an interactive **Streamlit** web application for single and batch predictions.

## 🚀 Features

- Predict the winning response between two LLM outputs.
- Supports three classes:
  - Model A
  - Model B
  - Tie
- TF-IDF based text feature extraction.
- Logistic Regression classifier.
- Batch prediction using CSV files.
- Download prediction results as CSV.
- Interactive Streamlit interface.

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- Joblib

## 📂 Project Structure

```
├── app.py
├── train_model.py
├── model_utils.py
├── model.joblib
├── train.csv
├── test.csv
├── submission.csv
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/llm-response-classifier.git
cd llm-response-classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Train the Model

```bash
python train_model.py
```

This will:
- Train the classifier
- Save the trained model
- Generate a submission file

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

## 📊 Workflow

1. Load training and test datasets.
2. Parse prompts and responses.
3. Convert text into TF-IDF features.
4. Train a Logistic Regression classifier.
5. Predict probabilities for:
   - Model A
   - Model B
   - Tie
6. Export predictions as CSV.

## 📈 Model

- TF-IDF Vectorizer
- Logistic Regression
- Validation using Log Loss

## Future Improvements

- Fine-tune Transformer models (BERT/RoBERTa)
- Ensemble learning
- Better feature engineering
- Hyperparameter optimization
- Explainable AI (SHAP/LIME)

## Author

**Chirag Vyas**

M.Tech – Artificial Intelligence & Data Science  
IIIT Kota

## License

This project is licensed under the MIT License.

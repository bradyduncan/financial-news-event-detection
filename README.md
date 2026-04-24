# Financial News Event Detection

## Setup

### Windows (PowerShell)
1. Create and install venv + deps:
```powershell
.\scripts\setup_venv.ps1
```
2. Activate later:
```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS
1. Create and install venv + deps:
```bash
bash scripts/setup_venv.sh
```
2. Activate later:
```bash
source .venv/bin/activate
```

### Linux
1. Create and install venv + deps:
```bash
bash scripts/setup_venv.sh
```
2. Activate later:
```bash
source .venv/bin/activate
```

### Manual (all platforms)
```bash
python -m venv .venv
```
```bash
python -m pip install -r requirements.txt
```

## Prepare Data

Run:
```powershell
python preprocessing/prepare_data.py
```

Common options:
```powershell
python preprocessing/prepare_data.py --subset sentences_75agree --test-size 0.2 --seed 42
python preprocessing/prepare_data.py --lemmatize
python preprocessing/prepare_data.py --no-negation
python preprocessing/prepare_data.py --ngram-min 1 --ngram-max 3
python preprocessing/prepare_data.py --max-features 20000
python preprocessing/prepare_data.py --output-dir artifacts
```

Arguments:
- `--subset`: Financial PhraseBank subset (e.g., `sentences_allagree`, `sentences_75agree`)
- `--test-size`: Test set proportion (default: 0.2)
- `--seed`: Random seed for train/test split
- `--lemmatize`: Enable lemmatization
- `--negation` / `--no-negation`: Toggle negation handling
- `--ngram-min`: Minimum n-gram size
- `--ngram-max`: Maximum n-gram size
- `--max-features`: Limit TF-IDF vocabulary size
- `--output-dir`: Directory to save TF-IDF artifacts (pipeline + train/test splits)


For a specific subset:
```bash
python preprocessing/prepare_data.py --subset sentences_75agree
```

Common options:
```bash
python preprocessing/prepare_data.py --subset sentences_75agree --test-size 0.2 --seed 42
python preprocessing/prepare_data.py --lemmatize
python preprocessing/prepare_data.py --no-negation
python preprocessing/prepare_data.py --ngram-min 1 --ngram-max 3
python preprocessing/prepare_data.py --max-features 20000
python preprocessing/prepare_data.py --output-dir artifacts
```

Arguments:
- `--subset`: Financial PhraseBank subset (e.g., `sentences_allagree`, `sentences_75agree`)
- `--test-size`: Test set proportion (default: 0.2)
- `--seed`: Random seed for train/test split
- `--lemmatize`: Enable lemmatization
- `--negation` / `--no-negation`: Toggle negation handling
- `--ngram-min`: Minimum n-gram size
- `--ngram-max`: Maximum n-gram size
- `--max-features`: Limit TF-IDF vocabulary size
- `--output-dir`: Directory to save TF-IDF artifacts (pipeline + train/test splits)

---

## Exploratory Data Analysis

Run the EDA script to analyze label distribution, sentence lengths, top words, financial keyword frequency, number/percentage presence, and sample sentences:

```bash
python analysis/eda.py
```

Output is printed to the terminal. No files are generated — findings are used to inform handcrafted feature engineering.

---

## Baseline Model (TF-IDF + Classical ML)

Train and evaluate Logistic Regression, Linear SVM, and Random Forest with GridSearchCV on TF-IDF + handcrafted features:

```bash
python pipelines/tf-idf_pipeline/train_evaluate.py --subset sentences_75agree
```

This will:
- Load TF-IDF matrices and combine with 8 handcrafted features
- Train three classifiers with 5-fold stratified cross-validation
- Evaluate on the test set (accuracy, macro-F1, ROC-AUC, per-class metrics, confusion matrix)
- Extract top Logistic Regression feature coefficients for interpretability
- Save trained models and results

Results saved to: `pipelines/tf-idf_pipeline/results/sentences_75agree/`

### Analyze Baseline Results

View summary metrics, best model, per-class F1, and generate confusion matrix charts:

```bash
python pipelines/tf-idf_pipeline/analyze_results.py --subset sentences_75agree --results-dir pipelines/tf-idf_pipeline/results
```

Output includes:
- Results summary table (accuracy, macro-F1, ROC-AUC per model)
- Best model by macro-F1
- Per-class F1 scores for each model
- Confusion matrix heatmaps saved as PNGs

Confusion matrix PNGs saved to: `pipelines/tf-idf_pipeline/results/sentences_75agree/charts/`

To analyze other subsets:
```bash
python pipelines/tf-idf_pipeline/analyze_results.py --subset sentences_50agree --results-dir pipelines/tf-idf_pipeline/results
python pipelines/tf-idf_pipeline/analyze_results.py --subset sentences_66agree --results-dir pipelines/tf-idf_pipeline/results
python pipelines/tf-idf_pipeline/analyze_results.py --subset sentences_allagree --results-dir pipelines/tf-idf_pipeline/results
```

### Run on All Subsets

```bash
python preprocessing/prepare_data.py --subset sentences_allagree
python pipelines/tf-idf_pipeline/train_evaluate.py --subset sentences_allagree

python preprocessing/prepare_data.py --subset sentences_75agree
python pipelines/tf-idf_pipeline/train_evaluate.py --subset sentences_75agree

python preprocessing/prepare_data.py --subset sentences_66agree
python pipelines/tf-idf_pipeline/train_evaluate.py --subset sentences_66agree

python preprocessing/prepare_data.py --subset sentences_50agree
python pipelines/tf-idf_pipeline/train_evaluate.py --subset sentences_50agree
```

---

## FinBERT Model

```bash
python pipelines/finbert_pipeline/finbert_pipeline.py --subset sentences_75agree
```

Results saved to: `pipelines/finbert_pipeline/results/sentences_75agree/`

---

## Per-Class Comparison

### Static Charts (PNG)
Generate comparison bar charts and confusion matrix heatmaps:

```bash
python analysis/comparison.py
```

Charts saved to: `analysis/dashboard_charts/`

### Interactive Dashboard (HTML)
Generate a self-contained HTML dashboard with interactive charts:

```bash
python analysis/generate_dashboard.py
```

Open `analysis/dashboard_charts/dashboard.html` in any browser. No server required.

---


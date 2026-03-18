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

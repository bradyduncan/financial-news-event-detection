"""Generate an interactive HTML comparison dashboard.
The dashboard displays a web page that visualizes the performance of the best 
TF-IDF and FinBERT models on the selected subset, using charts and tables to 
highlight differences in metrics and confusion matrices. 
The dashboard is designed to be visually appealing and easy to interpret,
making it ideal for presentations and visualizing differences between the two approaches."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

TFIDF_RESULTS = REPO_ROOT / "pipelines" / "tf-idf_pipeline" / "results"
FINBERT_RESULTS = REPO_ROOT / "pipelines" / "finbert_pipeline" / "results"
OUTPUT_DIR = REPO_ROOT / "analysis" / "dashboard_charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSET = "sentences_75agree"
LABEL_NAMES = ["negative", "neutral", "positive"]


def load_results(base_dir, subset):
    json_path = base_dir / subset / "results.json"
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_best_model(results):
    if results is None:
        return None
    best = None
    best_f1 = -1
    for entry in results.get("models", []):
        f1 = entry.get("metrics", {}).get("macro_f1", 0)
        if f1 > best_f1:
            best_f1 = f1
            best = entry
    return best


def build_html(tfidf, finbert):
    t = tfidf["metrics"] if tfidf else {}
    f = finbert["metrics"] if finbert else {}

    t_f1 = t.get("per_class", {}).get("f1", [0, 0, 0])
    f_f1 = f.get("per_class", {}).get("f1", [0, 0, 0])
    t_prec = t.get("per_class", {}).get("precision", [0, 0, 0])
    f_prec = f.get("per_class", {}).get("precision", [0, 0, 0])
    t_rec = t.get("per_class", {}).get("recall", [0, 0, 0])
    f_rec = f.get("per_class", {}).get("recall", [0, 0, 0])
    t_cm = t.get("confusion_matrix", [[0]*3]*3)
    f_cm = f.get("confusion_matrix", [[0]*3]*3)

    # Convert CM to percentages
    def cm_pct(cm):
        result = []
        for row in cm:
            total = sum(row)
            result.append([round(v / total, 2) if total > 0 else 0 for v in row])
        return result

    t_cm_pct = cm_pct(t_cm)
    f_cm_pct = cm_pct(f_cm)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TF-IDF vs FinBERT — Comparison Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background: #0a1628;
        color: #e0e7ef;
        padding: 20px;
    }}
    h1 {{
        text-align: center;
        font-size: 2rem;
        margin-bottom: 8px;
        background: linear-gradient(135deg, #4C72B0, #DD8452);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .subtitle {{
        text-align: center;
        color: #8899aa;
        margin-bottom: 30px;
        font-size: 0.95rem;
    }}
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 30px;
    }}
    .metric-card {{
        background: #111d30;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #1e2d44;
    }}
    .metric-card .label {{
        font-size: 0.8rem;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }}
    .metric-card .values {{
        display: flex;
        justify-content: center;
        gap: 20px;
    }}
    .metric-card .value {{
        font-size: 1.8rem;
        font-weight: bold;
    }}
    .metric-card .model-label {{
        font-size: 0.7rem;
        color: #8899aa;
    }}
    .tfidf-color {{ color: #4C72B0; }}
    .finbert-color {{ color: #DD8452; }}
    .charts-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-bottom: 30px;
    }}
    .chart-card {{
        background: #111d30;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #1e2d44;
    }}
    .chart-card h3 {{
        font-size: 1rem;
        margin-bottom: 15px;
        color: #c0ccdd;
    }}
    .full-width {{
        grid-column: 1 / -1;
    }}
    .cm-container {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }}
    .cm-box h4 {{
        text-align: center;
        margin-bottom: 10px;
        font-size: 0.9rem;
        color: #c0ccdd;
    }}
    table.cm {{
        width: 100%;
        border-collapse: collapse;
        text-align: center;
    }}
    table.cm th, table.cm td {{
        padding: 10px;
        font-size: 0.85rem;
        border: 1px solid #1e2d44;
    }}
    table.cm th {{
        background: #1a2940;
        color: #8899aa;
        font-weight: 600;
    }}
    .cm-tfidf td {{ background: rgba(76, 114, 176, 0.15); }}
    .cm-finbert td {{ background: rgba(221, 132, 82, 0.15); }}
    .cm-diag {{ font-weight: bold; }}
    .cm-tfidf .cm-diag {{ color: #4C72B0; background: rgba(76, 114, 176, 0.3); }}
    .cm-finbert .cm-diag {{ color: #DD8452; background: rgba(221, 132, 82, 0.3); }}
    .insights {{
        background: #111d30;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #1e2d44;
        margin-bottom: 20px;
    }}
    .insights h3 {{
        font-size: 1rem;
        margin-bottom: 12px;
        color: #c0ccdd;
    }}
    .insights ul {{
        list-style: none;
        padding: 0;
    }}
    .insights li {{
        padding: 8px 0;
        border-bottom: 1px solid #1e2d44;
        font-size: 0.9rem;
        color: #a0b0c0;
    }}
    .insights li:last-child {{ border-bottom: none; }}
    .insights .highlight {{ color: #e0e7ef; font-weight: 600; }}
    .footer {{
        text-align: center;
        color: #556677;
        font-size: 0.8rem;
        margin-top: 30px;
    }}
    @media (max-width: 768px) {{
        .charts-grid {{ grid-template-columns: 1fr; }}
        .cm-container {{ grid-template-columns: 1fr; }}
    }}
</style>
</head>
<body>

<h1>TF-IDF vs FinBERT</h1>
<p class="subtitle">Financial Sentiment Analysis — {SUBSET} subset</p>

<!-- Summary Metrics -->
<div class="metrics-grid">
    <div class="metric-card">
        <div class="label">Macro F1</div>
        <div class="values">
            <div><div class="value tfidf-color">{t.get('macro_f1', 0):.2f}</div><div class="model-label">TF-IDF</div></div>
            <div><div class="value finbert-color">{f.get('macro_f1', 0):.2f}</div><div class="model-label">FinBERT</div></div>
        </div>
    </div>
    <div class="metric-card">
        <div class="label">Accuracy</div>
        <div class="values">
            <div><div class="value tfidf-color">{t.get('accuracy', 0):.2f}</div><div class="model-label">TF-IDF</div></div>
            <div><div class="value finbert-color">{f.get('accuracy', 0):.2f}</div><div class="model-label">FinBERT</div></div>
        </div>
    </div>
    <div class="metric-card">
        <div class="label">ROC-AUC</div>
        <div class="values">
            <div><div class="value tfidf-color">{t.get('roc_auc_ovr', 0):.2f}</div><div class="model-label">TF-IDF</div></div>
            <div><div class="value finbert-color">{f.get('roc_auc_ovr', 0):.2f}</div><div class="model-label">FinBERT</div></div>
        </div>
    </div>
    <div class="metric-card">
        <div class="label">Best Model</div>
        <div class="values">
            <div><div class="value tfidf-color" style="font-size:1rem;">{tfidf['name'] if tfidf else 'N/A'}</div><div class="model-label">TF-IDF</div></div>
            <div><div class="value finbert-color" style="font-size:1rem;">{finbert['name'] if finbert else 'N/A'}</div><div class="model-label">FinBERT</div></div>
        </div>
    </div>
</div>

<!-- Charts -->
<div class="charts-grid">
    <div class="chart-card">
        <h3>Per-Class F1 Score</h3>
        <canvas id="f1Chart"></canvas>
    </div>
    <div class="chart-card">
        <h3>Per-Class Precision</h3>
        <canvas id="precChart"></canvas>
    </div>
    <div class="chart-card">
        <h3>Per-Class Recall</h3>
        <canvas id="recChart"></canvas>
    </div>
    <div class="chart-card">
        <h3>Overall Metrics</h3>
        <canvas id="overallChart"></canvas>
    </div>
</div>

<!-- Confusion Matrices -->
<div class="chart-card full-width" style="margin-bottom:20px;">
    <h3>Confusion Matrices (proportions)</h3>
    <div class="cm-container">
        <div class="cm-box">
            <h4 class="tfidf-color">TF-IDF ({tfidf['name'] if tfidf else 'N/A'})</h4>
            <table class="cm cm-tfidf">
                <tr><th></th><th>Pred Neg</th><th>Pred Neu</th><th>Pred Pos</th></tr>
                <tr><th>Neg</th><td class="cm-diag">{t_cm_pct[0][0]}</td><td>{t_cm_pct[0][1]}</td><td>{t_cm_pct[0][2]}</td></tr>
                <tr><th>Neu</th><td>{t_cm_pct[1][0]}</td><td class="cm-diag">{t_cm_pct[1][1]}</td><td>{t_cm_pct[1][2]}</td></tr>
                <tr><th>Pos</th><td>{t_cm_pct[2][0]}</td><td>{t_cm_pct[2][1]}</td><td class="cm-diag">{t_cm_pct[2][2]}</td></tr>
            </table>
        </div>
        <div class="cm-box">
            <h4 class="finbert-color">FinBERT ({finbert['name'] if finbert else 'N/A'})</h4>
            <table class="cm cm-finbert">
                <tr><th></th><th>Pred Neg</th><th>Pred Neu</th><th>Pred Pos</th></tr>
                <tr><th>Neg</th><td class="cm-diag">{f_cm_pct[0][0]}</td><td>{f_cm_pct[0][1]}</td><td>{f_cm_pct[0][2]}</td></tr>
                <tr><th>Neu</th><td>{f_cm_pct[1][0]}</td><td class="cm-diag">{f_cm_pct[1][1]}</td><td>{f_cm_pct[1][2]}</td></tr>
                <tr><th>Pos</th><td>{f_cm_pct[2][0]}</td><td>{f_cm_pct[2][1]}</td><td class="cm-diag">{f_cm_pct[2][2]}</td></tr>
            </table>
        </div>
    </div>
</div>

<!-- Key Insights -->
<div class="insights">
    <h3>Key Insights</h3>
    <ul>
        <li>FinBERT outperforms TF-IDF across <span class="highlight">all metrics and all classes</span></li>
        <li>Largest gap on <span class="highlight">negative class</span> — F1: {t_f1[0]:.2f} vs {f_f1[0]:.2f} (Δ {f_f1[0] - t_f1[0]:.2f})</li>
        <li>Smallest gap on <span class="highlight">neutral class</span> — F1: {t_f1[1]:.2f} vs {f_f1[1]:.2f} (Δ {f_f1[1] - t_f1[1]:.2f})</li>
        <li>TF-IDF baseline offers <span class="highlight">full interpretability</span> through feature coefficients and trains in minutes on CPU</li>
        <li>FinBERT's contextual understanding captures sentiment from <span class="highlight">word combinations</span> that TF-IDF treats independently</li>
    </ul>
</div>

<div class="footer">
    Financial News Sentiment Analysis — CS 6120 NLP — Kirtan Patel | Brady Duncan
</div>

<script>
const labels = {json.dumps(LABEL_NAMES)};
const tfidfColor = 'rgba(76, 114, 176, 0.85)';
const finbertColor = 'rgba(221, 132, 82, 0.85)';
const gridColor = 'rgba(255,255,255,0.08)';
const tickColor = '#8899aa';

Chart.defaults.color = tickColor;

function makeBarChart(id, label, tData, fData) {{
    new Chart(document.getElementById(id), {{
        type: 'bar',
        data: {{
            labels: labels,
            datasets: [
                {{ label: 'TF-IDF', data: tData.map(v => +v.toFixed(2)), backgroundColor: tfidfColor, borderRadius: 6 }},
                {{ label: 'FinBERT', data: fData.map(v => +v.toFixed(2)), backgroundColor: finbertColor, borderRadius: 6 }}
            ]
        }},
        options: {{
            responsive: true,
            plugins: {{
                legend: {{ position: 'top' }},
                tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2) }} }}
            }},
            scales: {{
                y: {{ min: 0, max: 1.05, grid: {{ color: gridColor }} }},
                x: {{ grid: {{ display: false }} }}
            }}
        }}
    }});
}}

makeBarChart('f1Chart', 'F1', {json.dumps(t_f1)}, {json.dumps(f_f1)});
makeBarChart('precChart', 'Precision', {json.dumps(t_prec)}, {json.dumps(f_prec)});
makeBarChart('recChart', 'Recall', {json.dumps(t_rec)}, {json.dumps(f_rec)});

new Chart(document.getElementById('overallChart'), {{
    type: 'bar',
    data: {{
        labels: ['Macro F1', 'Accuracy', 'ROC-AUC'],
        datasets: [
            {{ label: 'TF-IDF', data: [{t.get('macro_f1',0):.2f}, {t.get('accuracy',0):.2f}, {t.get('roc_auc_ovr',0):.2f}], backgroundColor: tfidfColor, borderRadius: 6 }},
            {{ label: 'FinBERT', data: [{f.get('macro_f1',0):.2f}, {f.get('accuracy',0):.2f}, {f.get('roc_auc_ovr',0):.2f}], backgroundColor: finbertColor, borderRadius: 6 }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{ position: 'top' }},
            tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2) }} }}
        }},
        scales: {{
            y: {{ min: 0, max: 1.05, grid: {{ color: gridColor }} }},
            x: {{ grid: {{ display: false }} }}
        }}
    }}
}});
</script>

</body>
</html>"""
    return html


def main():
    print("Loading results...")
    tfidf_data = load_results(TFIDF_RESULTS, SUBSET)
    finbert_data = load_results(FINBERT_RESULTS, SUBSET)

    tfidf_best = get_best_model(tfidf_data)
    finbert_best = get_best_model(finbert_data)

    if not tfidf_best and not finbert_best:
        print("No results found. Make sure both pipelines have been run.")
        return

    print("Generating HTML dashboard...")
    html = build_html(tfidf_best, finbert_best)

    output_path = OUTPUT_DIR / "dashboard.html"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard saved to: {output_path}")
    print(f"Open in browser: file://{output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
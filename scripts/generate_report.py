import json
from pathlib import Path
import base64
from src.utils.logger import get_logger, success

def generate_html_report(metrics_dir, viz_dir, output_path, logger=None):
    logger = logger or get_logger()
    metrics_dir = Path(metrics_dir)
    viz_dir = Path(viz_dir)
    output_path = Path(output_path)

    # Load metrics
    metrics = {}
    for tag in ['source', 'adapted', 'final']:
        path = metrics_dir / f"{tag}_metrics.json"
        if path.exists():
            with open(path) as f:
                metrics[tag] = json.load(f)

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lane Detection Evaluation Report</title>
        <style>
            table { border-collapse: collapse; width: 80%; margin: 20px auto; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
            h1, h2 { text-align: center; }
            .container { width: 90%; margin: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Lane Detection Model Performance Report</h1>
            <h2>Summary Table</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Source-only</th>
                    <th>Adapted</th>
                    <th>Final</th>
                </tr>
    """
    metrics_list = ['iou', 'f1', 'brier', 'pixel_entropy', 'fp_rate', 'fn_rate']
    display_names = ['mIoU (%)', 'F1 Score (%)', 'Brier Score', 'Pixel Entropy', 'FP Rate (lane)', 'FN Rate (lane)']
    for metric_name, display_name in zip(metrics_list, display_names):
        html += f"<tr><td>{display_name}</td>"
        for key in ['source', 'adapted', 'final']:
            if key in metrics:
                val = metrics[key].get(metric_name, 'N/A')
                if metric_name in ['iou', 'f1'] and isinstance(val, (int, float)):
                    val = f"{val*100:.2f}"
                elif isinstance(val, (int, float)):
                    val = f"{val:.4f}"
                html += f"<td>{val}</td>"
            else:
                html += "<td>N/A</td>"
        html += "</tr>"

    html += """
            </table>
            <h2>Sample Predictions</h2>
    """
    # Embed visualizations
    images = sorted(viz_dir.glob("*_sample_*.png"))
    for img_path in images[:4]:
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        html += f'<img src="data:image/png;base64,{img_data}" style="width:100%; max-width:800px; margin:10px auto; display:block;" />'

    html += """
        </div>
    </body>
    </html>
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    success(logger, "Report generated at %s", output_path)
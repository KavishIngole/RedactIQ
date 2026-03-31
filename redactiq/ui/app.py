"""Gradio UI for RedactIQ demo.

Provides an interactive web interface to demonstrate PII redaction,
show side-by-side comparisons of different modes, analytics dashboard
with anomaly visualization, and file-level redaction with same-format download.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
from collections import Counter
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go

from redactiq.redaction.pipeline import RedactionPipeline
from redactiq.utils.config import load_config
from redactiq.utils.file_parser import extract_text
from redactiq.utils.file_writer import (
    write_redacted_csv,
    write_redacted_docx,
    write_redacted_pdf,
    write_redacted_text,
    write_redacted_xlsx,
)


_pipeline: RedactionPipeline | None = None

# Supported file extensions for upload
_FILE_TYPES = [
    ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".csv", ".txt", ".log", ".json", ".tsv", ".md",
]


def _get_pipeline() -> RedactionPipeline:
    global _pipeline
    if _pipeline is None:
        config = load_config()
        config.setdefault("detection", {}).setdefault("llm", {})["enabled"] = False
        _pipeline = RedactionPipeline(config)
    return _pipeline


# ---------------------------------------------------------------------------
# Core redaction functions
# ---------------------------------------------------------------------------


def redact_text(text: str, mode: str, detect_anomalies: bool) -> tuple[str, str, str, str]:
    """Process text and return (redacted_text, entities_json, anomalies_json, stats)."""
    pipeline = _get_pipeline()
    result = pipeline.process(text=text, mode=mode, detect_anomalies=detect_anomalies)

    entities_info = [
        {
            "type": e.entity_type.value,
            "text": e.text,
            "confidence": round(e.confidence, 3),
            "source": e.source.value,
            "position": f"{e.start}-{e.end}",
        }
        for e in result.entities
    ]

    anomalies_info = [
        {
            "segment": a.segment_text[:100],
            "score": round(a.anomaly_score, 3),
            "reason": a.reason,
        }
        for a in result.anomaly_flags
    ]

    stats = (
        f"Processing: {result.processing_time_ms:.1f}ms | "
        f"Entities: {len(result.entities)} "
        f"(rules: {result.rule_detections}, LLM: {result.llm_detections}) | "
        f"Anomalies: {len(result.anomaly_flags)}"
    )

    return (
        result.redacted_text,
        json.dumps(entities_info, indent=2),
        json.dumps(anomalies_info, indent=2) if anomalies_info else "No anomalies detected",
        stats,
    )


def compare_modes(text: str) -> tuple[str, str, str]:
    """Show redaction result in all three modes side by side."""
    pipeline = _get_pipeline()
    # Detect once, redact thrice — avoids 3x detection overhead
    results = pipeline.detect_and_redact_multi(
        text, modes=["mask", "pseudonymize", "hash"]
    )
    return results["mask"], results["pseudonymize"], results["hash"]


def redact_file(
    file_obj, mode: str, detect_anomalies: bool
) -> tuple[str | None, str, str, str]:
    """Process an uploaded file and return redacted output IN THE SAME FORMAT.

    Returns (output_file_path, redacted_preview, entities_json, stats).
    """
    if file_obj is None:
        return None, "", "[]", "No file uploaded"

    pipeline = _get_pipeline()
    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    filename = Path(file_path).name
    ext = Path(filename).suffix.lower()

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    parsed = extract_text(file_bytes, filename)
    file_format = parsed["format"]

    all_entities: list[dict] = []
    total_anomalies = 0

    if file_format == "csv":
        # CSV: cell-by-cell redaction
        content = parsed["pages"][0]["text"]
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        redacted_rows = []

        for row in rows:
            redacted_row = []
            for cell in row:
                cell_text = cell.strip()
                if not cell_text:
                    redacted_row.append(cell)
                    continue
                redacted, cell_entities = pipeline.redact_cell(cell_text, mode=mode)
                redacted_row.append(redacted)
                for e in cell_entities:
                    all_entities.append({
                        "type": e.entity_type.value, "text": e.text,
                        "confidence": round(e.confidence, 3),
                    })
            redacted_rows.append(redacted_row)

        # Write same-format CSV
        out_ext = ".csv"
        out_path = Path(tempfile.gettempdir()) / f"redacted_{Path(filename).stem}{out_ext}"
        write_redacted_csv(redacted_rows, str(out_path))

        preview_rows = redacted_rows[:10]
        preview = "\n".join(" | ".join(c[:50] for c in r) for r in preview_rows)
        if len(redacted_rows) > 10:
            preview += f"\n... ({len(redacted_rows) - 10} more rows)"

    else:
        # PDF, DOCX, XLSX, TXT etc: page-by-page redaction
        redacted_pages = []  # For file writer: [{original_text, redacted_text, label}]
        preview_sections = []

        for page_info in parsed["pages"]:
            page_text = page_info["text"]
            if not page_text.strip():
                continue

            result = pipeline.process(
                text=page_text, mode=mode, detect_anomalies=detect_anomalies
            )
            label = page_info["page"]
            redacted_pages.append({
                "original_text": page_text,
                "redacted_text": result.redacted_text,
                "label": str(label),
            })
            preview_sections.append(f"--- {label} ---\n{result.redacted_text}")

            for e in result.entities:
                all_entities.append({
                    "type": e.entity_type.value, "text": e.text,
                    "confidence": round(e.confidence, 3), "page": str(label),
                })
            total_anomalies += len(result.anomaly_flags)

        # Write output in the SAME format as input
        out_ext = ext or ".txt"
        out_path = Path(tempfile.gettempdir()) / f"redacted_{Path(filename).stem}{out_ext}"

        if file_format == "docx" and ext in (".docx", ".doc"):
            try:
                write_redacted_docx(file_bytes, redacted_pages, str(out_path))
            except Exception:
                out_path = out_path.with_suffix(".txt")
                write_redacted_text("\n\n".join(preview_sections), str(out_path))
        elif file_format == "xlsx" and ext in (".xlsx", ".xls"):
            try:
                write_redacted_xlsx(file_bytes, redacted_pages, str(out_path))
            except Exception:
                out_path = out_path.with_suffix(".txt")
                write_redacted_text("\n\n".join(preview_sections), str(out_path))
        elif file_format == "pdf":
            try:
                write_redacted_pdf(redacted_pages, str(out_path))
            except Exception:
                out_path = out_path.with_suffix(".txt")
                write_redacted_text("\n\n".join(preview_sections), str(out_path))
        else:
            out_path = out_path.with_suffix(ext if ext else ".txt")
            write_redacted_text("\n\n".join(preview_sections), str(out_path))

        full_preview = "\n\n".join(preview_sections)
        preview = full_preview[:3000]
        if len(full_preview) > 3000:
            preview += "\n... (truncated)"

    page_count = len(parsed["pages"])
    stats = (
        f"File: {filename} ({file_format.upper()}) | "
        f"Sections: {page_count} | "
        f"Entities: {len(all_entities)} | "
        f"Anomalies: {total_anomalies} | "
        f"Output: {out_path.suffix}"
    )

    return str(out_path), preview, json.dumps(all_entities, indent=2), stats


# ---------------------------------------------------------------------------
# Analytics charting
# ---------------------------------------------------------------------------

_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]

_CHART_TEMPLATE = "plotly_white"


def _empty_fig(title: str = "") -> go.Figure:
    """Return a placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(
        text="Run analysis to see results", xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#aaa"),
    )
    fig.update_layout(
        title=title, xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=320, template=_CHART_TEMPLATE,
    )
    return fig


def _build_entity_charts(entities: list, text: str, source_label: str = "text input"):
    """Build 4 entity-related charts + summary."""
    type_counts = Counter(e.entity_type.value for e in entities)
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    types, counts = zip(*sorted_types) if sorted_types else ([], [])

    # 1. Entity type bar chart
    fig_bar = go.Figure(go.Bar(
        x=list(counts), y=list(types), orientation="h",
        marker_color=_COLORS[:len(types)],
        text=list(counts), textposition="outside",
    ))
    fig_bar.update_layout(
        title="PII Entity Type Distribution", xaxis_title="Count",
        yaxis=dict(autorange="reversed"),
        height=max(280, len(types) * 38 + 80),
        template=_CHART_TEMPLATE, margin=dict(l=140, r=50, t=40, b=30),
    )

    # 2. Confidence histogram
    fig_hist = go.Figure(go.Histogram(
        x=[round(e.confidence, 2) for e in entities], nbinsx=20,
        marker_color="#636EFA", marker_line=dict(color="white", width=1),
    ))
    fig_hist.update_layout(
        title="Confidence Score Distribution",
        xaxis_title="Confidence", yaxis_title="Count",
        height=320, template=_CHART_TEMPLATE, margin=dict(l=50, r=30, t=40, b=30),
    )

    # 3. Detection source donut
    source_counts = Counter(e.source.value for e in entities)
    fig_pie = go.Figure(go.Pie(
        labels=list(source_counts.keys()), values=list(source_counts.values()),
        hole=0.45, marker_colors=["#00CC96", "#EF553B", "#AB63FA"],
        textinfo="label+percent+value",
    ))
    fig_pie.update_layout(
        title="Detection Source", height=320, template=_CHART_TEMPLATE,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # 4. Risk heatmap
    conf_buckets = ["0.70-0.80", "0.80-0.85", "0.85-0.90", "0.90-0.95", "0.95-1.00"]
    bucket_ranges = [(0.70, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
    unique_types = sorted(set(e.entity_type.value for e in entities))
    heatmap_data = []
    for etype in unique_types:
        type_ents = [e for e in entities if e.entity_type.value == etype]
        heatmap_data.append([
            sum(1 for e in type_ents if lo <= e.confidence < hi)
            for lo, hi in bucket_ranges
        ])

    fig_heat = go.Figure(go.Heatmap(
        z=heatmap_data, x=conf_buckets, y=unique_types,
        colorscale="YlOrRd", text=heatmap_data, texttemplate="%{text}",
        showscale=True, colorbar_title="Count",
    ))
    fig_heat.update_layout(
        title="Risk Heatmap (Type x Confidence)", xaxis_title="Confidence Range",
        height=max(280, len(unique_types) * 32 + 80),
        template=_CHART_TEMPLATE, margin=dict(l=140, r=50, t=40, b=30),
    )

    # Summary
    total = len(entities)
    top = sorted_types[0] if sorted_types else ("N/A", 0)
    avg_conf = sum(e.confidence for e in entities) / total
    high_conf = sum(1 for e in entities if e.confidence >= 0.95)
    chars_pii = sum(len(e.text) for e in entities)
    pct = (chars_pii / max(len(text), 1)) * 100

    summary = (
        f"Source: {source_label}\n"
        f"Total PII: {total} entities across {len(type_counts)} types\n"
        f"Most common: {top[0]} ({top[1]}x)\n"
        f"Confidence: avg {avg_conf:.2f} | high (>=0.95): {high_conf}/{total}\n"
        f"PII characters: {chars_pii} ({pct:.1f}% of input)"
    )

    return fig_bar, fig_hist, fig_pie, fig_heat, summary


def _build_anomaly_chart(anomaly_flags: list) -> go.Figure:
    """Build anomaly visualization from anomaly flags."""
    if not anomaly_flags:
        fig = go.Figure()
        fig.add_annotation(
            text="No anomalies detected", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#2ca02c"),
        )
        fig.update_layout(
            title="Anomaly Detection Results",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            height=280, template=_CHART_TEMPLATE,
        )
        return fig

    segments = [a.segment_text[:40] + "..." if len(a.segment_text) > 40
                else a.segment_text for a in anomaly_flags]
    scores = [round(a.anomaly_score, 3) for a in anomaly_flags]
    reasons = [a.reason for a in anomaly_flags]

    colors = ["#EF553B" if s > 0.8 else "#FFA15A" if s > 0.5 else "#636EFA" for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=segments, orientation="h",
        marker_color=colors,
        text=[f"{s:.2f} - {r[:30]}" for s, r in zip(scores, reasons)],
        textposition="outside",
        hovertext=[f"Score: {s}<br>Reason: {r}<br>Text: {seg}"
                   for s, r, seg in zip(scores, reasons,
                                        [a.segment_text[:80] for a in anomaly_flags])],
        hoverinfo="text",
    ))
    fig.update_layout(
        title=f"Anomaly Detection ({len(anomaly_flags)} flagged)",
        xaxis_title="Anomaly Score", xaxis=dict(range=[0, 1.1]),
        yaxis=dict(autorange="reversed"),
        height=max(250, len(anomaly_flags) * 40 + 80),
        template=_CHART_TEMPLATE, margin=dict(l=200, r=120, t=40, b=30),
    )
    return fig


def analyze_text(text: str, mode: str):
    """Run PII + anomaly analysis on text and return 5 charts + summary."""
    if not text.strip():
        e = _empty_fig()
        return e, e, e, e, e, "No text provided"

    pipeline = _get_pipeline()
    result = pipeline.process(text=text, mode=mode, detect_anomalies=True)

    if not result.entities and not result.anomaly_flags:
        e = _empty_fig("No PII or Anomalies Detected")
        return e, e, e, e, e, "No PII entities or anomalies found."

    if result.entities:
        fig_bar, fig_hist, fig_pie, fig_heat, summary = _build_entity_charts(
            result.entities, text
        )
    else:
        fig_bar = fig_hist = fig_pie = fig_heat = _empty_fig("No PII Detected")
        summary = "No PII entities found."

    fig_anomaly = _build_anomaly_chart(result.anomaly_flags)

    return fig_bar, fig_hist, fig_pie, fig_heat, fig_anomaly, summary


def analyze_file(file_obj, mode: str):
    """Run PII + anomaly analysis on a file and return 5 charts + summary."""
    if file_obj is None:
        e = _empty_fig()
        return e, e, e, e, e, "No file uploaded"

    pipeline = _get_pipeline()
    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    filename = Path(file_path).name

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    parsed = extract_text(file_bytes, filename)
    all_entities = []
    all_anomalies = []
    full_text = ""

    if parsed["format"] == "csv":
        content = parsed["pages"][0]["text"]
        reader = csv.reader(io.StringIO(content))
        for row in reader:
            for cell in row:
                cell_text = cell.strip()
                if not cell_text:
                    continue
                _, cell_entities = pipeline.redact_cell(cell_text, mode=mode)
                all_entities.extend(cell_entities)
                full_text += cell_text + " "
    else:
        for page_info in parsed["pages"]:
            page_text = page_info["text"]
            if not page_text.strip():
                continue
            result = pipeline.process(text=page_text, mode=mode, detect_anomalies=True)
            all_entities.extend(result.entities)
            all_anomalies.extend(result.anomaly_flags)
            full_text += page_text + " "

    if not all_entities and not all_anomalies:
        e = _empty_fig("No PII or Anomalies Detected")
        return e, e, e, e, e, f"No PII or anomalies found in {filename}"

    if all_entities:
        fig_bar, fig_hist, fig_pie, fig_heat, summary = _build_entity_charts(
            all_entities, full_text, filename
        )
    else:
        fig_bar = fig_hist = fig_pie = fig_heat = _empty_fig("No PII Detected")
        summary = f"No PII found in {filename}"

    fig_anomaly = _build_anomaly_chart(all_anomalies)

    return fig_bar, fig_hist, fig_pie, fig_heat, fig_anomaly, summary


# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------

_SAMPLES = [
    (
        "John Smith's email is john.smith@acme.com and his phone is "
        "(555) 123-4567. He lives at 123 Oak Street, Springfield, IL 62701. "
        "SSN: 123-45-6789."
    ),
    (
        "Patient record - Aadhaar: 9876 5432 1012, PAN: ABCPD1234E, "
        "Phone: +91 98765 43210, UPI: patient@paytm, "
        "Voter ID: ABC1234567, Passport No: J8369854."
    ),
    (
        "Employee: Maria Garcia, SSN 987-65-4321, email m.garcia@techcorp.com. "
        "GSTIN: 27AAPFU0939F1ZV. Bank IFSC: SBIN0001234. IP: 192.168.1.100."
    ),
]


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------

_CSS = """
.main-title { text-align: center; margin-bottom: 0; }
.subtitle { text-align: center; color: #666; font-size: 0.95em; margin-top: 0; }
.section-header { color: #1a56db; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; }
"""


def build_ui() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(
        title="RedactIQ - Smart PII Redaction",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
        css=_CSS,
    ) as demo:
        gr.Markdown(
            "<h1 class='main-title'>RedactIQ</h1>"
            "<p class='subtitle'>Smart PII Redaction Assistant &mdash; "
            "Hybrid Detection (Rules + Qwen3-8B) &bull; Anomaly Detection &bull; "
            "21 PII Patterns (India + Global) &bull; Multi-Format Support</p>"
        )

        # ---- Tab 1: Text Redaction -----------------------------------------
        with gr.Tab("Redact Text"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown("#### Input", elem_classes=["section-header"])
                    input_text = gr.Textbox(
                        label="Paste text containing PII",
                        lines=8,
                        placeholder="Enter text with sensitive data...",
                    )
                    with gr.Row():
                        mode = gr.Dropdown(
                            choices=["mask", "pseudonymize", "hash"],
                            value="mask", label="Mode",
                        )
                        anomaly_toggle = gr.Checkbox(label="Anomaly Detection", value=False)
                    redact_btn = gr.Button("Redact", variant="primary", size="lg")

                    gr.Markdown("##### Quick Samples")
                    with gr.Row():
                        for i, sample in enumerate(_SAMPLES):
                            gr.Button(f"Sample {i+1}", size="sm").click(
                                fn=lambda s=sample: s, outputs=input_text,
                            )

                with gr.Column(scale=1):
                    gr.Markdown("#### Output", elem_classes=["section-header"])
                    output_text = gr.Textbox(label="Redacted Text", lines=8)
                    stats_text = gr.Textbox(label="Stats", lines=1, interactive=False)
                    with gr.Accordion("Detected Entities", open=False):
                        entities_json = gr.Code(language="json", lines=8)
                    with gr.Accordion("Anomaly Flags", open=False):
                        anomalies_json = gr.Code(language="json", lines=5)

            redact_btn.click(
                fn=redact_text,
                inputs=[input_text, mode, anomaly_toggle],
                outputs=[output_text, entities_json, anomalies_json, stats_text],
            )

        # ---- Tab 2: Compare Modes ------------------------------------------
        with gr.Tab("Compare Modes"):
            gr.Markdown("#### Side-by-side comparison of all redaction modes")
            compare_input = gr.Textbox(label="Input Text", lines=4, value=_SAMPLES[1])
            compare_btn = gr.Button("Compare All Modes", variant="primary")
            with gr.Row():
                mask_out = gr.Textbox(label="Mask (*****)", lines=5)
                pseudo_out = gr.Textbox(label="Pseudonymize (fake data)", lines=5)
                hash_out = gr.Textbox(label="Hash ([HASH:abc...])", lines=5)
            compare_btn.click(
                fn=compare_modes, inputs=[compare_input],
                outputs=[mask_out, pseudo_out, hash_out],
            )

        # ---- Tab 3: File Upload + Same-format Download ----------------------
        with gr.Tab("File Redaction"):
            gr.Markdown(
                "#### Upload a document to redact PII\n"
                "Supported: **PDF, DOCX, XLSX, CSV, TXT, LOG, JSON, MD, TSV**. "
                "Output is downloaded **in the same format** as your input."
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload File", file_types=_FILE_TYPES)
                    with gr.Row():
                        file_mode = gr.Dropdown(
                            choices=["mask", "pseudonymize", "hash"],
                            value="mask", label="Mode",
                        )
                        file_anomaly = gr.Checkbox(label="Anomaly Detection", value=False)
                    file_btn = gr.Button("Redact File", variant="primary", size="lg")
                with gr.Column(scale=1):
                    file_output = gr.File(label="Download Redacted File")
                    file_stats = gr.Textbox(label="Stats", lines=1, interactive=False)
                    with gr.Accordion("Preview", open=True):
                        file_preview = gr.Textbox(label="Redacted Content", lines=10)
                    with gr.Accordion("Detected Entities", open=False):
                        file_entities = gr.Code(language="json", lines=6)

            file_btn.click(
                fn=redact_file,
                inputs=[file_input, file_mode, file_anomaly],
                outputs=[file_output, file_preview, file_entities, file_stats],
            )

        # ---- Tab 4: Analytics Dashboard ------------------------------------
        with gr.Tab("Analytics"):
            gr.Markdown(
                "#### PII Analytics Dashboard\n"
                "Visualize entity distribution, confidence levels, detection sources, "
                "risk heatmap, and **anomaly detection results**."
            )

            with gr.Tab("From Text"):
                analytics_text = gr.Textbox(
                    label="Input Text", lines=5, value=_SAMPLES[1],
                    placeholder="Paste text to analyze...",
                )
                with gr.Row():
                    a_mode = gr.Dropdown(
                        choices=["mask", "pseudonymize", "hash"],
                        value="mask", label="Mode",
                    )
                    a_btn = gr.Button("Analyze", variant="primary")

                with gr.Row():
                    c_bar = gr.Plot(label="Entity Distribution")
                    c_pie = gr.Plot(label="Detection Source")
                with gr.Row():
                    c_hist = gr.Plot(label="Confidence Histogram")
                    c_heat = gr.Plot(label="Risk Heatmap")
                with gr.Row():
                    c_anomaly = gr.Plot(label="Anomaly Detection")
                a_summary = gr.Textbox(label="Summary", lines=5, interactive=False)

                a_btn.click(
                    fn=analyze_text, inputs=[analytics_text, a_mode],
                    outputs=[c_bar, c_hist, c_pie, c_heat, c_anomaly, a_summary],
                )

            with gr.Tab("From File"):
                a_file = gr.File(label="Upload File", file_types=_FILE_TYPES)
                with gr.Row():
                    af_mode = gr.Dropdown(
                        choices=["mask", "pseudonymize", "hash"],
                        value="mask", label="Mode",
                    )
                    af_btn = gr.Button("Analyze File", variant="primary")

                with gr.Row():
                    fc_bar = gr.Plot(label="Entity Distribution")
                    fc_pie = gr.Plot(label="Detection Source")
                with gr.Row():
                    fc_hist = gr.Plot(label="Confidence Histogram")
                    fc_heat = gr.Plot(label="Risk Heatmap")
                with gr.Row():
                    fc_anomaly = gr.Plot(label="Anomaly Detection")
                af_summary = gr.Textbox(label="Summary", lines=5, interactive=False)

                af_btn.click(
                    fn=analyze_file, inputs=[a_file, af_mode],
                    outputs=[fc_bar, fc_hist, fc_pie, fc_heat, fc_anomaly, af_summary],
                )

        # ---- Tab 5: About --------------------------------------------------
        with gr.Tab("About"):
            gr.Markdown("""
## How It Works

```
Input (Text / PDF / DOCX / XLSX / CSV)
    |
    v
[Rule-Based Detection] --- 21 regex patterns (India + Global PII)
    |
    v
[Qwen3-8B LLM] ---------- Context-aware entity detection via remote API
    |
    v
[Hybrid Merger] ---------- Union / Intersection / Priority strategies
    |
    v
[Redaction Engine] ------- Mask (*) / Pseudonymize (fake data) / Hash (SHA-256)
    |
    v
[Anomaly Detection] ------ Flags disguised or obfuscated PII via LLM
    |
    v
Output (same-format file + analytics + anomaly report)
```

## Supported PII Types

| Global | India-Specific |
|--------|---------------|
| Email, Phone, SSN | Aadhaar (Verhoeff validated) |
| Credit Card (Luhn) | PAN Card |
| IP Address, Passport | IFSC Code, GSTIN |
| Date of Birth, DL | Indian Phone (+91) |
| Organization, Address | Voter ID, Indian DL, UPI ID |

## Key Features
- **Hybrid Detection**: Rules catch structured PII; LLM catches contextual PII
- **Anomaly Detection**: Catches disguised/obfuscated PII patterns
- **Same-Format Output**: Upload DOCX, get back DOCX. Upload PDF, get back PDF.
- **Analytics Dashboard**: Interactive charts for PII distribution and risk assessment
- **Intel Optimized**: vLLM with AMX/BF16 on Intel Xeon CPUs
            """)

    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()

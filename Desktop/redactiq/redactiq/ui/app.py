"""Gradio UI for RedactIQ demo.

Provides an interactive web interface to demonstrate PII redaction,
show side-by-side comparisons of different modes, and display
anomaly detection results.
"""

from __future__ import annotations

import json

import gradio as gr

from redactiq.redaction.pipeline import RedactionPipeline
from redactiq.utils.config import load_config


_pipeline: RedactionPipeline | None = None


def _get_pipeline() -> RedactionPipeline:
    global _pipeline
    if _pipeline is None:
        config = load_config()
        # Disable LLM for lightweight demo (can be toggled)
        config.setdefault("detection", {}).setdefault("llm", {})["enabled"] = False
        _pipeline = RedactionPipeline(config)
    return _pipeline


def redact_text(text: str, mode: str, detect_anomalies: bool) -> tuple[str, str, str, str]:
    """Process text and return (redacted_text, entities_json, anomalies_json)."""
    pipeline = _get_pipeline()
    result = pipeline.process(
        text=text,
        mode=mode,
        detect_anomalies=detect_anomalies,
    )

    entities_info = []
    for e in result.entities:
        entities_info.append({
            "type": e.entity_type.value,
            "text": e.text,
            "confidence": round(e.confidence, 3),
            "source": e.source.value,
            "position": f"{e.start}-{e.end}",
        })

    anomalies_info = []
    for a in result.anomaly_flags:
        anomalies_info.append({
            "segment": a.segment_text[:100],
            "score": round(a.anomaly_score, 3),
            "reason": a.reason,
        })

    stats = (
        f"Processing time: {result.processing_time_ms:.1f}ms | "
        f"Entities found: {len(result.entities)} "
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
    results = {}
    for mode in ["mask", "pseudonymize", "hash"]:
        result = pipeline.process(text=text, mode=mode, detect_anomalies=False)
        results[mode] = result.redacted_text
    return results["mask"], results["pseudonymize"], results["hash"]


# Sample texts for demo
_SAMPLES = [
    (
        "John Smith's email is john.smith@acme.com and his phone is "
        "(555) 123-4567. He lives at 123 Oak Street, Springfield, IL 62701. "
        "SSN: 123-45-6789. Credit card: 4532-1234-5678-9012."
    ),
    (
        "Patient Jane Doe (DOB: 03/15/1985) was admitted to Memorial Hospital. "
        "Contact: jane.doe@email.com, (312) 555-0198. "
        "Insurance ID: BCBS-9876543. MRN: MRN-00123456. "
        "Referring physician: Dr. Robert Wilson."
    ),
    (
        "Employee record: Maria Garcia, SSN 987-65-4321, hired at "
        "TechCorp Inc. Office: 456 Innovation Dr, San Jose, CA 95134. "
        "Work email: m.garcia@techcorp.com. IP: 192.168.1.100."
    ),
]


def build_ui() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(
        title="RedactIQ - Smart PII Redaction",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# RedactIQ - Smart PII Redaction Assistant")
        gr.Markdown(
            "Hybrid PII detection (rules + LLM) with anomaly detection, "
            "optimized for Intel Xeon CPUs via vLLM."
        )

        with gr.Tab("Redact"):
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Input Text",
                        lines=8,
                        placeholder="Paste text containing PII here...",
                    )
                    with gr.Row():
                        mode = gr.Dropdown(
                            choices=["mask", "pseudonymize", "hash"],
                            value="mask",
                            label="Redaction Mode",
                        )
                        anomaly_toggle = gr.Checkbox(
                            label="Detect Anomalies",
                            value=True,
                        )
                    redact_btn = gr.Button("Redact", variant="primary")

                    gr.Markdown("### Sample Texts")
                    for i, sample in enumerate(_SAMPLES):
                        gr.Button(f"Sample {i+1}", size="sm").click(
                            fn=lambda s=sample: s,
                            outputs=input_text,
                        )

                with gr.Column():
                    output_text = gr.Textbox(label="Redacted Output", lines=8)
                    entities_json = gr.Code(
                        label="Detected Entities",
                        language="json",
                        lines=10,
                    )
                    anomalies_json = gr.Code(
                        label="Anomaly Flags",
                        language="json",
                        lines=5,
                    )
                    stats_text = gr.Textbox(label="Processing Stats", lines=1)

            redact_btn.click(
                fn=redact_text,
                inputs=[input_text, mode, anomaly_toggle],
                outputs=[output_text, entities_json, anomalies_json, stats_text],
            )

        with gr.Tab("Compare Modes"):
            gr.Markdown("Compare all three redaction modes side by side.")
            compare_input = gr.Textbox(
                label="Input Text",
                lines=5,
                value=_SAMPLES[0],
            )
            compare_btn = gr.Button("Compare", variant="primary")
            with gr.Row():
                mask_output = gr.Textbox(label="Mask Mode", lines=5)
                pseudo_output = gr.Textbox(label="Pseudonymize Mode", lines=5)
                hash_output = gr.Textbox(label="Hash Mode", lines=5)

            compare_btn.click(
                fn=compare_modes,
                inputs=[compare_input],
                outputs=[mask_output, pseudo_output, hash_output],
            )

        with gr.Tab("About"):
            gr.Markdown("""
## Architecture

```
Input Text
    |
    v
[Rule-Based Detection] --> Regex patterns for emails, phones, SSNs, etc.
    |
    v
[LLM Detection] ---------> Context-aware entity detection via vLLM
    |
    v
[Hybrid Merger] ----------> Combines results (union/intersection/priority)
    |
    v
[Redaction Engine] -------> Mask / Pseudonymize / Hash
    |
    v
[Anomaly Detection] ------> Flags unusual patterns (Isolation Forest)
    |
    v
Output: Redacted text + Entity list + Anomaly flags
```

## Key Features
- **Hybrid Detection**: Rules catch structured PII; LLM catches contextual PII
- **Anomaly Detection**: Catches disguised/obfuscated PII patterns
- **Intel Optimized**: vLLM with AMX/BF16 on Intel Xeon CPUs
- **Multiple Modes**: Mask, pseudonymize, or hash PII
- **Audit Trail**: Full logging for compliance
            """)

    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()

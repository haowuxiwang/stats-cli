"""Comprehensive analysis report generation."""

import os
from datetime import datetime


def report(values, usl=None, lsl=None, target=None, **kwargs):
    """Generate comprehensive analysis report.

    Args:
        values: Data values
        usl: Upper specification limit
        lsl: Lower specification limit
        target: Target value

    Returns:
        Dict with comprehensive analysis results
    """
    from stats_engine.descriptive import descriptive
    from stats_engine.normality import normality
    from stats_engine.outlier import outlier

    result = {
        "analyses": {},
    }

    # Descriptive statistics
    result["analyses"]["descriptive"] = descriptive(values)

    # Normality test
    result["analyses"]["normality"] = normality(values)

    # Outlier detection
    result["analyses"]["outlier"] = outlier(values, method="grubbs")

    # Capability (if spec limits provided)
    if usl is not None or lsl is not None:
        from stats_engine.capability import capability

        result["analyses"]["capability"] = capability(values, usl=usl, lsl=lsl, target=target)

    # Control chart
    from stats_engine.control_chart import control_chart

    result["analyses"]["control_chart"] = control_chart("imr", values)

    # Summary
    desc = result["analyses"]["descriptive"]
    norm = result["analyses"]["normality"]
    result["summary"] = {
        "n": desc["n"],
        "mean": desc["mean"],
        "std": desc["std"],
        "is_normal": norm["is_normal"],
        "n_outliers": result["analyses"]["outlier"]["n_outliers"],
    }

    if "capability" in result["analyses"]:
        cap = result["analyses"]["capability"]
        result["summary"]["cpk"] = cap.get("cpk")
        result["summary"]["rating"] = cap.get("rating")

    return result


def export_excel(report_data, output_path="report.xlsx", **kwargs):
    """Export report to Excel format.

    Args:
        report_data: Report data from report() function
        output_path: Output file path

    Returns:
        Dict with export result
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas and openpyxl required for Excel export")

    # Create Excel writer
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Summary sheet
        if "summary" in report_data:
            summary_df = pd.DataFrame([report_data["summary"]])
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Descriptive statistics
        if "analyses" in report_data and "descriptive" in report_data["analyses"]:
            desc = report_data["analyses"]["descriptive"]
            desc_df = pd.DataFrame([desc])
            desc_df.to_excel(writer, sheet_name="Descriptive", index=False)

        # Normality test
        if "analyses" in report_data and "normality" in report_data["analyses"]:
            norm = report_data["analyses"]["normality"]
            norm_data = {}
            for key, val in norm.items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        norm_data[f"{key}_{k}"] = v
                else:
                    norm_data[key] = val
            norm_df = pd.DataFrame([norm_data])
            norm_df.to_excel(writer, sheet_name="Normality", index=False)

        # Capability (if present)
        if "analyses" in report_data and "capability" in report_data["analyses"]:
            cap = report_data["analyses"]["capability"]
            cap_df = pd.DataFrame([cap])
            cap_df.to_excel(writer, sheet_name="Capability", index=False)

        # Outlier detection
        if "analyses" in report_data and "outlier" in report_data["analyses"]:
            outlier = report_data["analyses"]["outlier"]
            outlier_df = pd.DataFrame([outlier])
            outlier_df.to_excel(writer, sheet_name="Outlier", index=False)

    return {
        "format": "excel",
        "output_path": output_path,
        "file_size": os.path.getsize(output_path),
        "sheets": ["Summary", "Descriptive", "Normality", "Capability", "Outlier"],
    }


def _register_cjk_font(pdf):
    """Try to register a CJK font for Chinese/Japanese/Korean text.
    Returns font name to use, or 'Helvetica' if no CJK font found.
    """
    import platform
    from pathlib import Path

    font_paths = []
    system = platform.system()
    if system == "Windows":
        font_paths = [
            ("simhei", "C:/Windows/Fonts/simhei.ttf"),
            ("msyh", "C:/Windows/Fonts/msyh.ttc"),
        ]
    elif system == "Darwin":
        font_paths = [
            ("heiti", "/System/Library/Fonts/STHeiti Medium.ttc"),
            ("arialuni", "/Library/Fonts/Arial Unicode.ttf"),
        ]
    else:
        font_paths = [
            ("wqy", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
            ("noto", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ]

    for name, path in font_paths:
        if Path(path).exists():
            try:
                pdf.add_font(name, "", path)
                pdf.add_font(name, "B", path)
                return name
            except Exception:
                continue
    return "Helvetica"


def export_pdf(report_data, output_path="report.pdf", **kwargs):
    """Export report to PDF format.

    Args:
        report_data: Report data from report() function
        output_path: Output file path

    Returns:
        Dict with export result
    """
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
    except ImportError:
        raise ImportError("fpdf2 required for PDF export. Install with: pip install fpdf2")

    pdf = FPDF()
    pdf.add_page()

    # Register CJK font for Chinese/Japanese/Korean text support
    font = _register_cjk_font(pdf)

    # Title
    pdf.set_font(font, "B", 16)
    pdf.cell(0, 10, "Statistical Analysis Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(5)

    # Timestamp
    pdf.set_font(font, "", 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)

    # Summary section
    if "summary" in report_data:
        pdf.set_font(font, "B", 14)
        pdf.cell(0, 10, "Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(font, "", 11)

        summary = report_data["summary"]
        for key, value in summary.items():
            if value is not None:
                pdf.cell(0, 8, f"{key}: {value}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

    # Descriptive statistics
    if "analyses" in report_data and "descriptive" in report_data["analyses"]:
        pdf.set_font(font, "B", 14)
        pdf.cell(0, 10, "Descriptive Statistics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(font, "", 11)

        desc = report_data["analyses"]["descriptive"]
        for key in ["n", "mean", "std", "rsd_percent", "min", "max", "range", "q1", "q3", "iqr"]:
            if key in desc and desc[key] is not None:
                pdf.cell(0, 8, f"{key}: {desc[key]}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

    # Normality test
    if "analyses" in report_data and "normality" in report_data["analyses"]:
        pdf.set_font(font, "B", 14)
        pdf.cell(0, 10, "Normality Test", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(font, "", 11)

        norm = report_data["analyses"]["normality"]
        pdf.cell(0, 8, f"Is Normal: {norm.get('is_normal', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if "shapiro_wilk" in norm:
            sw = norm["shapiro_wilk"]
            pdf.cell(0, 8, f"Shapiro-Wilk p-value: {sw.get('p_value', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

    # Capability (if present)
    if "analyses" in report_data and "capability" in report_data["analyses"]:
        pdf.set_font(font, "B", 14)
        pdf.cell(0, 10, "Process Capability", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(font, "", 11)

        cap = report_data["analyses"]["capability"]
        for key in ["cp", "cpk", "pp", "ppk", "rating"]:
            if key in cap and cap[key] is not None:
                pdf.cell(0, 8, f"{key}: {cap[key]}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

    # Outlier detection
    if "analyses" in report_data and "outlier" in report_data["analyses"]:
        pdf.set_font(font, "B", 14)
        pdf.cell(0, 10, "Outlier Detection", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(font, "", 11)

        outlier = report_data["analyses"]["outlier"]
        pdf.cell(0, 8, f"Number of outliers: {outlier.get('n_outliers', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

    # Save PDF
    pdf.output(output_path)

    return {
        "format": "pdf",
        "output_path": output_path,
        "file_size": os.path.getsize(output_path),
    }

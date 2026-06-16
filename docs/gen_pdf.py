"""Generate skill architecture PDF document."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos


class SkillPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, "stats-cli-py Skill Architecture", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
            self.line(10, 20, 200, 20)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(5)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(41, 128, 185)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(240, 240, 240)
        self.multi_cell(0, 5, text, fill=True)
        self.ln(3)

    def table_row(self, cells, widths, bold=False):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 9)
        for i, (cell, w) in enumerate(zip(cells, widths)):
            self.set_fill_color(245, 245, 245) if i % 2 == 0 else self.set_fill_color(255, 255, 255)
            self.cell(w, 7, str(cell)[:int(w/2)], border=1, fill=True)
        self.ln()


def main():
    pdf = SkillPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # === Cover ===
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 15, "stats-cli-py", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "AI Agent Skill Architecture Document", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "How We Built It | Best Practices | AI Agent Integration", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(20)
    pdf.set_draw_color(41, 128, 185)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "Version: 1.2.1", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.cell(0, 8, "Date: 2026-06-15", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.cell(0, 8, "Commands: 33 | Tests: 1033 | Coverage: 95%", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    # === TOC ===
    pdf.add_page()
    pdf.chapter_title("Table of Contents")
    toc = [
        ("1. Skill Overview", "What is stats-cli-py?"),
        ("2. Architecture", "Three-layer design"),
        ("3. How AI Agents Call It", "JSON dispatcher pattern"),
        ("4. Progressive Disclosure", "Two-layer design"),
        ("5. Best Practices Score", "8 criteria evaluation"),
        ("6. Decision Trees", "Statistical methodology routing"),
        ("7. Self-Discovery", "The discover command"),
        ("8. Improvement History", "What we improved"),
    ]
    for title, desc in toc:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(80, 8, title)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)

    # === Ch1: Overview ===
    pdf.add_page()
    pdf.chapter_title("1. Skill Overview")
    pdf.body_text(
        "stats-cli-py is a pure Python statistical analysis CLI/library designed for "
        "manufacturing and quality engineering. It provides 33 statistical commands covering "
        "descriptive statistics, hypothesis testing, SPC, capability analysis, DOE, MSA, "
        "reliability, multivariate analysis, time series, and more."
    )
    pdf.body_text("Key Design Principles:")
    pdf.body_text("  - All I/O is JSON: stdin/stdout or file-based")
    pdf.body_text("  - Pure Python: powered by scipy + statsmodels, no R dependency")
    pdf.body_text("  - AI-Agent native: self-describing API with discover command")
    pdf.body_text("  - Progressive disclosure: simple entry, deep on demand")
    pdf.ln(5)
    pdf.section_title("Command Categories (33 commands)")
    cats = [
        ("Basic", "descriptive, normality, outlier"),
        ("Hypothesis", "ttest, anova, nonparametric, homogeneity, equivalence, power"),
        ("SPC", "control_chart, capability, trend"),
        ("Regression", "regression, correlation"),
        ("DOE", "doe (full_factorial, fractional, taguchi, RSM)"),
        ("MSA", "gage_rr (crossed, nested, attribute)"),
        ("Reliability", "reliability (weibull, kaplan_meier)"),
        ("Multivariate", "multivariate (pca, cluster, discriminant)"),
        ("Time Series", "timeseries (exp_smoothing, arima, decomposition)"),
        ("Data", "explore, clean, transform, discover"),
        ("Workflow", "workflow, workflow_template, check_assumptions, recommend"),
        ("Report", "report, export_excel, export_pdf, run"),
    ]
    w = [40, 150]
    pdf.table_row(["Category", "Commands"], w, bold=True)
    for cat, cmds in cats:
        pdf.table_row([cat, cmds], w)

    # === Ch2: Architecture ===
    pdf.add_page()
    pdf.chapter_title("2. Architecture")
    pdf.body_text("The skill uses a three-layer architecture:")
    pdf.ln(3)
    pdf.section_title("Layer 1: Router (main.py)")
    pdf.body_text(
        "Parses JSON input, routes commands via _route() function with lazy imports, "
        "handles exceptions, wraps output in standard envelope. 374 lines."
    )
    pdf.ln(3)
    pdf.section_title("Layer 2: Engine (stats_engine/*.py)")
    pdf.body_text(
        "28 independent modules, each exposing one public function that accepts **params "
        "and returns a dict. All computation uses scipy.stats, numpy, and statsmodels. "
        "No custom statistical implementations."
    )
    pdf.ln(3)
    pdf.section_title("Layer 3: Utils (utils/*.py)")
    pdf.body_text(
        "4 utility modules: data_loader (file I/O with encoding detection), "
        "data_cleaner (NaN/Inf handling), validators (input validation), "
        "output (JSON envelope with rounding)."
    )
    pdf.ln(5)
    pdf.section_title("Data Flow")
    pdf.code_block(
        "Input JSON -> main.py handler() -> _route(command, params)\n"
        "    -> stats_engine/module.function(**params)\n"
        "    -> utils/output.success(data) or error(msg)\n"
        "    -> JSON stdout"
    )

    # === Ch3: AI Agent Integration ===
    pdf.add_page()
    pdf.chapter_title("3. How AI Agents Call It")
    pdf.section_title("Invocation Patterns")
    pdf.code_block(
        '# Pattern 1: stdin pipe\n'
        "echo '{\"command\":\"descriptive\",\"params\":{\"values\":[1,2,3]}}' | python main.py\n\n"
        "# Pattern 2: file argument\n"
        "python main.py input.json\n\n"
        "# Pattern 3: Python import\n"
        "from main import handler\n"
        'result = handler({"command": "descriptive", "params": {"values": [1,2,3]}})'
    )
    pdf.ln(3)
    pdf.section_title("Standard Output Envelope")
    pdf.code_block(
        'Success: {"status":"success", "version":"1.2.1", "timestamp":"...", "data":{...}}\n'
        'Error:   {"status":"error", "error_type":"...", "message":"...", "suggestion":"..."}'
    )
    pdf.ln(3)
    pdf.section_title("Agent Call Flow (7 Steps)")
    pdf.body_text("1. Agent receives vague user request (\"analyze my data\")")
    pdf.body_text("2. Agent calls discover -> gets 33 command summaries")
    pdf.body_text("3. Agent uses description/category to select command")
    pdf.body_text("4. Agent calls discover {command_name:\"xxx\"} -> gets param schema")
    pdf.body_text("5. Agent constructs JSON request using example as template")
    pdf.body_text("6. Agent invokes command -> gets result")
    pdf.body_text("7. Agent uses output_fields to parse and interpret result")

    # === Ch4: Progressive Disclosure ===
    pdf.add_page()
    pdf.chapter_title("4. Progressive Disclosure")
    pdf.body_text("The skill implements TWO layers of progressive disclosure:")
    pdf.ln(3)
    pdf.section_title("Layer 1: SKILL.md Static Document (4 levels)")
    pdf.body_text("Level 1: Quick Start decision tree (5 top-level branches)")
    pdf.body_text("  -> Level 2: 4 detailed decision trees (comparison, relationship, prediction, QC)")
    pdf.body_text("    -> Level 3: 5 scenario-based workflows (complete JSON examples)")
    pdf.body_text("      -> Level 4: 33 command reference (all parameters)")
    pdf.ln(3)
    pdf.section_title("Layer 2: discover Runtime Dynamic Discovery (2 levels)")
    pdf.body_text("Level 1: discover (no params) -> lightweight summary (description + category)")
    pdf.body_text("  -> Level 2: discover {command_name:\"xxx\"} -> full schema (params + output_fields)")
    pdf.ln(5)
    pdf.section_title("Why Two Layers?")
    pdf.body_text(
        "Static SKILL.md is loaded once when the skill is activated. It provides decision "
        "trees and scenarios for common cases. The discover command is queried at runtime "
        "for edge cases not covered by SKILL.md, or when the agent needs exact parameter schemas."
    )

    # === Ch5: Best Practices ===
    pdf.add_page()
    pdf.chapter_title("5. Best Practices Score")
    w = [45, 20, 125]
    pdf.table_row(["Criterion", "Score", "Key Strength"], w, bold=True)
    scores = [
        ("Trigger Conditions", "Strong", "Bilingual triggers, broad coverage"),
        ("Progressive Disclosure", "M-Strong", "Two-layer design, decision trees"),
        ("Decision Guidance", "Strong", "4 decision trees, recommend command"),
        ("Error Handling", "Moderate", "7 error types, suggestion field"),
        ("Example-Driven", "Strong", "5 scenarios, 33 command examples"),
        ("Self-Documenting", "Strong", "Runtime discover with structured params"),
        ("Composition", "M-Strong", "workflow, 9 templates, auto_check"),
        ("Input Validation", "Strong", "Structured params with required/type"),
    ]
    for row in scores:
        pdf.table_row(row, w)
    pdf.ln(5)
    pdf.section_title("Key Design Wins")
    pdf.body_text("1. discover command: Runtime self-description, agents learn capabilities without reading docs")
    pdf.body_text("2. Decision trees: Encode correct statistical methodology (check normality first)")
    pdf.body_text("3. recommend command: Meta-guidance - agent can ask the tool what to use")
    pdf.body_text("4. workflow_template: 9 predefined templates for common analysis patterns")
    pdf.body_text("5. Structured params: required/type/desc for every parameter in discover")

    # === Ch6: Decision Trees ===
    pdf.add_page()
    pdf.chapter_title("6. Decision Trees")
    pdf.section_title("Comparison Analysis Tree")
    pdf.code_block(
        "User wants to compare data\n"
        "    |\n"
        "    +-- Only 2 groups?\n"
        "    |   +-- Normal? -> ttest (two_sample)\n"
        "    |   +-- Non-normal? -> nonparametric mann_whitney\n"
        "    |       +-- Also consider: transform boxcox -> retest\n"
        "    |\n"
        "    +-- 3+ groups?\n"
        "    |   +-- Normal? -> anova one_way\n"
        "    |   +-- Non-normal? -> nonparametric kruskal_wallis\n"
        "    |\n"
        "    +-- Paired data?\n"
        "        +-- Normal? -> ttest paired\n"
        "        +-- Non-normal? -> nonparametric wilcoxon"
    )
    pdf.ln(3)
    pdf.section_title("Quality Control Tree")
    pdf.code_block(
        "User wants quality control\n"
        "    |\n"
        "    +-- Monitor stability? -> control_chart (imr/xbar/r)\n"
        "    +-- Check capability? -> capability (needs USL/LSL)\n"
        "    +-- Measurement system? -> gage_rr (crossed/nested)\n"
        "    +-- Reliability? -> reliability (weibull/kaplan_meier)\n"
        "    +-- Detect trends? -> trend (cusum/ewma/runs)"
    )

    # === Ch7: Self-Discovery ===
    pdf.add_page()
    pdf.chapter_title("7. Self-Discovery (discover command)")
    pdf.body_text(
        "The discover command is the runtime self-description mechanism. It allows AI agents "
        "to learn about available commands without reading SKILL.md."
    )
    pdf.ln(3)
    pdf.section_title("Three Query Modes")
    pdf.code_block(
        "# Mode 1: List all commands (lightweight)\n"
        '{"command": "discover"}\n\n'
        "# Mode 2: Filter by category\n"
        '{"command": "discover", "params": {"category": "spc"}}\n\n'
        "# Mode 3: Single command details (full schema)\n"
        '{"command": "discover", "params": {"command_name": "capability"}}'
    )
    pdf.ln(3)
    pdf.section_title("Structured Parameters (v1.2.1)")
    pdf.body_text("Each command now exposes structured parameter metadata:")
    pdf.code_block(
        '"params": [\n'
        '  {"name": "values", "required": true, "type": "array", "desc": "Process data"},\n'
        '  {"name": "usl",    "required": true, "type": "number", "desc": "Upper spec limit"},\n'
        '  {"name": "lsl",    "required": true, "type": "number", "desc": "Lower spec limit"},\n'
        '  {"name": "target", "required": false, "type": "number", "desc": "Target value"}\n'
        "]"
    )
    pdf.body_text("This eliminates guesswork - agents know exactly which parameters are required vs optional.")

    # === Ch8: Improvements ===
    pdf.add_page()
    pdf.chapter_title("8. Improvement History")
    pdf.section_title("v1.2.1 Changes (2026-06-15)")
    pdf.body_text("SKILL.md Improvements:")
    pdf.body_text("  + Intent-to-Command quick lookup table (19 commands)")
    pdf.body_text("  + discover command recommendation at document top")
    pdf.body_text("  + Negative triggers (NOT for: streaming, text, image)")
    pdf.body_text("  + Top 5 command output examples")
    pdf.body_text("  + Error recovery workflow table")
    pdf.body_text("  + auto_check failure semantics documentation")
    pdf.body_text("  + run command sandbox restrictions")
    pdf.body_text("  + Box-Cox cross-links in decision trees")
    pdf.body_text("  + All placeholder [...] replaced with runnable data")
    pdf.ln(3)
    pdf.body_text("Code Improvements:")
    pdf.body_text("  + Structured params in discover.py (33 commands)")
    pdf.body_text("  + fpdf2 deprecation fixes (report.py)")
    pdf.body_text("  + Matplotlib deprecation fix (charts.py)")
    pdf.body_text("  + Exception logging instead of swallowing")
    pdf.ln(3)
    pdf.body_text("Packaging:")
    pdf.body_text("  + pyproject.toml: version 1.2.1, fpdf2 dependency, build-backend fix")
    pdf.body_text("  + Zip: excludes TODO.md, CLAUDE.md, tests, build artifacts")
    pdf.body_text("  + Final zip: 47 files, 310 KB")

    # Save
    pdf.output("D:/learn/claudecode/stats-cli/docs/skill-architecture.pdf")
    print("PDF generated: docs/skill-architecture.pdf")


if __name__ == "__main__":
    main()

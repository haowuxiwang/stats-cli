# stats-cli-py: Future Outlook & Strategic Assessment

**Date:** 2026-07-01
**Version:** 1.4.0
**Status:** Stable baseline, maintenance mode

---

## 1. Current State Summary

| Metric | Value |
|--------|-------|
| Commands | 39 (10 categories) |
| Test coverage | 96.24% |
| Tests | 1825 passed |
| CI matrix | ubuntu × windows × Python 3.10/3.11/3.12 |
| Avg response time | 0.022s (typical), 1.1s (100k points) |
| Distribution | 2 ZIP packages (Claude Code + Feishu Aily) |

---

## 2. JMP Capability Comparison

### 2.1 What JMP has that we don't (Gaps)

| JMP Feature | Our Status | Gap Severity |
|-------------|-----------|--------------|
| **Interactive GUI** (drag-and-drop, point-and-click) | Not applicable — we're CLI/JSON | By design |
| **Graph Builder** (interactive visualizations) | Static base64 PNG only | HIGH for interactive use |
| **Distribution platform** (real-time histogram brushing) | Single-shot distribution fit | MEDIUM |
| **Model screening** (automated model comparison) | Manual per-command calls | MEDIUM |
| **Definitive screening designs (DSD)** | Not implemented | MEDIUM |
| **Space-filling designs** | Not implemented | LOW |
| **Survival: Cox PH, Frailty, Competing Risks** | Only KM + Weibull + ALT | HIGH for pharma |
| **Gaussian Process / Kriging** | Not implemented | LOW |
| **Neural Network / Deep Learning** | Not implemented | LOW (out of scope) |
| **Text mining / NLP** | Not implemented | Out of scope |
| **Mixed models: Repeated measures, Split-plot** | Basic mixed effects only | MEDIUM |
| **Item analysis / IRT** | Not implemented | LOW |
| **Quality: Reliability growth, Duane, Crow-AMSAA** | Basic only | LOW |
| **Consumer research: Choice models, MaxDiff** | Not implemented | Out of scope |
| **Process screening: Screening designs** | Not implemented | LOW |
| **Automated EDA** (Explore outliers, patterns) | `explore` command covers basics | LOW |
| **JMP Scripting Language (JSL)** | `run` command (limited Python) | MEDIUM |
| **Live data connectivity** (ODBC, streaming) | File-based only | MEDIUM |
| **Dashboard / Report generation** | `report` + `export_excel/pdf` | LOW |

### 2.2 What we have that JMP doesn't (Differentiators)

| Our Advantage | Description |
|--------------|-------------|
| **AI-native design** | JSON-in/JSON-out, discoverable, self-documenting |
| **Programmatic access** | `handler()` function for any Python/AI integration |
| **Zero GUI overhead** | Pure Python, runs anywhere, no license |
| **Free and open source** | No $1,800/year license fee |
| **AI agent integration** | SKILL.md for Claude Code, Aily, Codex |
| **Reproducibility** | JSON inputs = exact reproduction of any analysis |
| **CI/CD friendly** | Headless, testable, version-controllable |
| **Cross-validation suite** | 17 tests proving R/JMP numerical accuracy |

### 2.3 Parity Matrix (JMP vs stats-cli-py)

| Analysis Domain | JMP Capability | Our Capability | Parity |
|----------------|---------------|----------------|--------|
| Descriptive statistics | Full | Full | 95% |
| Normality testing | Full | Full | 95% |
| t-tests | Full | Full | 95% |
| ANOVA (1-way, 2-way, N-way) | Full | 1-way + 2-way | 80% |
| Non-parametric tests | Full | Full | 90% |
| Regression (linear, logistic, penalized) | Full | Full | 85% |
| Regression (GLM, mixed, nonlinear) | Full | Basic GLM + mixed | 60% |
| Control charts (all types) | Full | Full | 90% |
| Process capability | Full | Full | 90% |
| Gage R&R (all types) | Full | Full | 85% |
| DOE (screening, response surface) | Full | Basic factorial + RSM | 65% |
| Reliability (Weibull, KM, ALT) | Full | Weibull + KM + ALT | 70% |
| Reliability (Cox PH, Frailty) | Full | Not implemented | 20% |
| Multivariate (PCA, clustering, discriminant) | Full | Full | 80% |
| Time series (ARIMA, ETS, decomposition) | Full | Basic ARIMA + ETS | 60% |
| Survival analysis | Full | KM + Weibull | 50% |
| Power analysis | Full | Full | 85% |
| Distribution fitting | Full | Full | 85% |
| Bayesian analysis | Basic | Basic | 70% |
| Data cleaning/transformation | Full | Full | 85% |
| Acceptance sampling | Full | Full | 85% |
| Sensitivity analysis | Full | Full | 80% |
| Functional data analysis | Not in JMP | Full | N/A |

**Overall estimated parity: ~75%** for core manufacturing/quality use cases.

---

## 3. Can Claude Code + This Skill Match JMP?

### Short answer: **For 80% of routine analyses, YES. For advanced/exploratory work, NO.**

### Where Claude Code + stats-cli-py WINS over JMP:

1. **Natural language interface** — "Compare these two batches" vs navigating menus
2. **Automated assumption checking** — `check_assumptions` + `recommend` does what JMP's "Fit Model" does manually
3. **Batch processing** — Analyze 100 files in a loop vs clicking through each
4. **Documentation generation** — `report` + `export_pdf` creates audit-ready reports automatically
5. **Integration with other AI tools** — Can chain with document generation, email, databases
6. **Cost** — Free vs $1,800/year per seat
7. **Reproducibility** — JSON inputs = exact reproduction, no "which button did I click?"

### Where JMP WINS over Claude Code + stats-cli-py:

1. **Interactive exploration** — Brushing histograms, clicking outliers, live linking
2. **Visual data discovery** — Graph Builder, Bubble Plot, Parallel Plot
3. **Advanced modeling** — Neural nets, Gaussian processes, choice models
4. **Real-time process monitoring** — Live data streaming, dashboards
5. **Regulatory submission** — JMP Clinical for FDA submissions
6. **User training** — GUI is easier for non-programmers
7. **Depth in niche areas** — Consumer research, reliability growth, spatial stats

### The "JMP Killer" Scenario:

For a **manufacturing quality engineer** who needs:
- Batch release decisions (capability + control chart)
- Gage R&R studies
- DOE for process optimization
- Stability trending
- OOS investigation

**Claude Code + stats-cli-py can replace 90% of their JMP usage** because:
- All these analyses are well-defined and repeatable
- JSON inputs ensure audit trail
- Reports auto-generate for batch records
- No license cost per seat

### The "JMP Required" Scenario:

For a **research statistician** who needs:
- Exploratory data analysis with visual brushing
- Advanced model comparison (neural nets, GAMs, etc.)
- Interactive presentation to stakeholders
- Regulatory submission (JMP Clinical)

**JMP remains essential** because:
- Visual interaction is core to the workflow
- Advanced models aren't in our scope
- Regulatory acceptance requires validated tools

---

## 4. Current Limitations

### 4.1 Technical Limitations

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| No interactive GUI | Can't explore data visually | Use chart_base64 + HTML viewer |
| Max ~100k samples in memory | Big data not supported | Sample or use file-based input |
| No live data streaming | Real-time monitoring not possible | Poll with scheduled runs |
| Single-threaded handler | No parallel processing within one call | Use ThreadPoolExecutor externally |
| No database connectivity | Must load files manually | Pre-load with Python script |
| `run` command security | Blacklist bypass possible | Remove `run` for production |
| No session state | Each call is independent | Use `workflow` for multi-step |

### 4.2 Functional Limitations

| Missing Feature | Priority | Effort |
|----------------|----------|--------|
| Cox Proportional Hazards | HIGH (pharma) | Medium |
| Repeated measures ANOVA | HIGH (stability) | Medium |
| N-way ANOVA (3+ factors) | MEDIUM | Low |
| Step-down/Step-up multiple testing | MEDIUM | Low |
| CUSUM with FIR (fast initial response) | MEDIUM | Low |
| Acceptance sampling: sequential plans | LOW | Medium |
| Reliability growth (Duane, AMSAA) | LOW | Medium |
| Measurement uncertainty (GUM) | MEDIUM | High |
| Bayesian: hierarchical models | LOW | High |
| Time series: VAR, VARMA, GARCH | LOW | High |

### 4.3 CI/CD Limitations

| Gap | Status | Recommendation |
|-----|--------|---------------|
| No automated PyPI publish | Manual release | Add GitHub Action on tag push |
| No Docker image | Not needed for Python skill | Optional for server deployment |
| No integration tests in CI | Unit tests only | Add e2e test job |
| No performance regression test | No baseline tracking | Add benchmark job with thresholds |
| No security scanning | `run` command risk | Add bandit/trivy scan |
| No notification on failure | Silent | Add Slack/email webhook |

---

## 5. Future Roadmap Recommendations

### Phase 1: Stabilize (Current — Maintenance Only)
- Fix bugs as reported
- Keep dependencies updated
- Monitor CI health
- **No new features**

### Phase 2: Close Critical Gaps (If Needed)
- Cox PH survival analysis (pharma requirement)
- Repeated measures (stability studies)
- N-way ANOVA (interaction effects)
- Measurement uncertainty (ISO 17025 requirement)

### Phase 3: Enhance AI Agent Experience
- Auto-generate analysis plan from natural language
- Smart data type detection (skip normality test if n < 3)
- Automatic report narrative generation
- Integration with LLM for result interpretation

### Phase 4: Scale (If Needed)
- Database connector (SQL queries → analysis)
- REST API wrapper (HTTP server mode)
- Dashboard generation (HTML with embedded charts)
- Multi-file batch processing with progress tracking

---

## 6. Recommendation

**For the current use case (AI agent skill for manufacturing/quality analysis):**

The tool is **production-ready** for:
- Batch release decisions
- SPC monitoring
- Gage R&R studies
- DOE analysis
- Stability trending
- OOS investigation
- Regulatory report generation

**It is NOT a JMP replacement** for:
- Exploratory data analysis
- Advanced modeling
- Interactive presentations
- Research statistics

**The optimal positioning is:**
> "An AI-native statistical analysis skill that handles 80% of routine manufacturing/quality analyses, designed to be called by AI agents rather than humans directly."

This is a **complement to JMP**, not a replacement. Use JMP for exploration and advanced modeling; use stats-cli-py for automated, reproducible, AI-driven analysis workflows.

---

## 7. Key Metrics Dashboard

```
Commands:        39 / ~200+ (JMP)     = 20% breadth
Core parity:     ~75% for mfg/quality
Test coverage:   96.24%                = excellent
Performance:     <1s typical           = excellent
Cost:            $0                    = vs $1,800/seat/year
AI integration:  Native               = unique differentiator
Maintenance:     Low (stable deps)     = sustainable
```

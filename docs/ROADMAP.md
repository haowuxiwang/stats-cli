# stats-cli-py: Strategic Roadmap

**Date:** 2026-07-02
**Current:** v1.4.0+phase3, 42 commands, 1894 tests, 96.08% coverage
**Position:** AI-native quality/SPC decision engine (NOT a JMP clone)

---

## 核心定位差异

| | JMP | stats-cli-py |
|---|-----|-------------|
| **交互** | GUI 拖拽探索 | JSON-in/JSON-out |
| **用户** | 统计学家手动分析 | AI Agent 自动批处理 |
| **优势** | 可视化、交互式 brushing | 可复现、CI/CD、零成本 |
| **模式** | "看看数据什么样" | "自动决策批放行/ Hold" |
| **许可** | $1,800/seat/年 | MIT 开源 |

**结论：不要对标 JMP 的功能广度，而是做 JMP 做不了的事。**

---

## Roadmap (维护模式)

### Phase A: 持续改进 (Ongoing)

**不做新功能，只修复和改善。**

| 触发条件 | 响应 |
|----------|------|
| Agent 用户报错 | 修复 → 测试 → push patch |
| 数值精度问题 | 添加 R 交叉验证测试 |
| CI 失败 | 立即修复 |
| 安全漏洞 | 24h 内响应 |

**质量标准:**
- 测试覆盖率 ≥ 93%
- 所有 example JSON 必须可执行
- 每次 push CI 全绿
- decision block 对齐检查

---

### Phase B: 按需扩展 (User-Driven)

**仅当真实用户（Agent）表现出需求时才添加。**

最高优先级 (如需求出现):

| 功能 | 原因 | 工作量 |
|------|------|--------|
| `regression` VIF / Cook's D | Agent 部署了错误模型 | 2h |
| `ttest` 正态性预检 | Agent 对非正态数据用 ttest 未警告 | 1h |
| `anova` 自动 post-hoc | Agent 不知道 ANOVA 显著后要做什么 | 2h |
| 图形控制面板 (Streamlit) | Agent 需要向人类展示交互式图表 | 8h |

**不做的 (JMP 能做但我们不应该做的):**

| 不做 | 原因 |
|------|------|
| t-SNE / UMAP | 无监督降维是 research 工具，非制造质量 |
| 深度学习 (CNN/RNN) | 与 scipy 统计 tool 定位不符 |
| 交互 GUI | 我们的优势是 Agent 可调用，不是人类点击 |
| 生存分析 frailty | Niche 需求，lifelines 库可用 `run` 实现 |

---

### Phase C: 差异化优势 (Differentiate)

**发挥 JMP 做不了的优势。**

| 方向 | 价值 | 状态 |
|------|------|------|
| **REST API** | CI/CD 集成，HTTP 调用 | ✅ `serve.py` done |
| **批处理工作流** | `workflow_template` 自动链 | ✅ done |
| **决策块** | Agent 无需判断即可行动 | ✅ 5 modules done |
| **自动假设检查** | `recommend` + `check_assumptions` | ✅ done |
| **PDF/Excel 报告** | `export_pdf` + `export_excel` | ✅ done |
| **飞书 Aily 集成** | SKILL.md + SKILL_AILY.md | ✅ done |
| **GitHub Actions** | npm/pip 风格 CI | ✅ CI matrix × 6 |

---

## 对标 JMP 覆盖度趋势

```
当前:     ████████████████████░░░░  ~82% (manufacturing/quality focus)
Phase B:  █████████████████████░  ~88% (按需补齐)
上限:     ██████████████████████  ~95% (永远不是 100%，因为我们不做 GUI)
```

**82% → 95% 的路径不是加功能，是让现有功能更可靠。**

---

## 维护 Runbook

### 每周
```bash
pytest tests/ -q --cov --cov-fail-under=93
pip list --outdated
```

### 每次合并
1. PR review (至少 1 人)
2. CI 全绿
3. examples/ 中的 JSON 全部验证
4. CHANGELOG 条目

### 每月
1. 依赖更新 (patch only, minor 评估, major 不开)
2. 对抗性审查 (spot-check 5 个命令)
3. User feedback triage

---

## 版本策略

| 版本 | 触发 |
|------|------|
| patch (1.4.x) | Bug fix, typo, warning suppression |
| minor (1.5.0) | New command, new decision block |
| major (2.0.0) | Breaking API change (avoid if possible) |

当前建议: **停留在 1.4.x patch 序列，除非有需要的新功能。**

# 项目总结 — stats-cli-py

**日期:** 2026-07-02
**版本:** v1.4.0 (Phase 1-3 完成)
**定位:** AI 原生质量/SPC 决策引擎（CLI + Skill）

---

## 经历了什么

### 三轮对抗性审查 (Adversarial Review)

| 轮次 | 焦点 | 发现 Bug | 修复 |
|------|------|----------|------|
| Round 1 | Agent 调用舒适度 + CI | 1 coverage 超限 | 修复 CI 矩阵 |
| Round 2 | 安全 + 输入验证 | 17 test failures | 安全加固 + 164 新测试 |
| Round 3 | 决策引擎 + 文档准确性 | 9 decision bug | 9/9 修复 |

### 四轮 Agent Team 并行执行

| Agent 任务 | 产出 |
|-----------|------|
| 覆盖率提升 | 95% → 96.08% |
| Phase 1 (生产就绪) | 安全、文档、CI、打包 |
| Phase 2 (对标缺口) | Cox PH、Auto-ARIMA、重复测量、CIs |
| Phase 3 (差异化) | REST API、决策块 |

---

## 当前指标

```
测试:       1894 passed, 0 failed, 10 skipped
覆盖率:     96.08%
Python 文件: 133
命令数:     40 (discover) + discover + run = 42 端点
文档:       14 个 md 文件
ZIP 包:     stats-cli-py.zip (180.5 KB)
            stats-cli-py-aily.zip (167.0 KB)
CI:         ubuntu × windows × Python 3.10/3.11/3.12
JMP 对标:   ~82% (制造/质量核心场景)
```

---

## 42 个端点覆盖

| 类别 | 数 | 命令 |
|------|---|------|
| Basic | 3 | descriptive, normality, outlier |
| Hypothesis | 7 | ttest, anova, nonparametric, homogeneity, multiple_comparison, equivalence, power |
| SPC | 4 | capability, control_chart, trend, acceptance_sampling |
| Regression | 2 | regression, correlation |
| Advanced | 8 | timeseries, advanced, run, distribution, bayesian, mining, sensitivity, functional |
| Data | 4 | explore, clean, transform, discover |
| Report | 3 | report, export_excel, export_pdf |
| Workflow | 4 | workflow, check_assumptions, recommend, workflow_template |
| 专项 | 7 | doe, gage_rr, multivariate, reliability, acceptance_sampling, measurement_uncertainty |

---

## 关键决策块 (Decision Blocks)

| 模块 | action 类型 | 包含 |
|------|-----------|------|
| capability | RELEASE / CONDITIONAL_RELEASE / HOLD | Cpk 值、阈值判断、批放行建议 |
| ttest (3) | REJECT_H0 / FAIL_TO_REJECT_H0 | p 值、Cohen's d、小样本降级 |
| control_chart (12) | RELEASE / HOLD / INVESTIGATE | Western Electric 规则违反、具体点位 |
| normality | NORMAL / NON_NORMAL | 推荐检验方法、样本量指南 |

---

## 文档体系

```
SKILL.md         ← AI Agent 入口 (触发条件、命令表、输出格式)
OUTPUT.md        ← 关键指标解读 (p值、Cpk、R²...)
OUTLOOK.md       ← 战略评估 (JMP 对标、局限、路线图)
CHANGELOG.md     ← 版本历史
CONTRIBUTING.md  ← 开发者指南
SECURITY.md      ← 安全模型
docs/ROADMAP.md  ← 维护模式策略
VALIDATION.md    ← 精度证明 (NIST + R 交叉验证)
```

---

## 维护模式已进入

**不再主动开发新功能。** 触发条件:

| 触发 | 响应 |
|------|------|
| Agent 报错 | 修复 → 测试 → push |
| CI 失败 | 立即修复 |
| 安全漏洞 | 24h 内响应 |
| 用户反馈 | 评估 → 排期 |

---
*Generated: 2026-07-02*

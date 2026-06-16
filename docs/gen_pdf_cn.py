"""生成 skill 架构中文 PDF 文档."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos


class SkillPDF(FPDF):
    def __init__(self):
        super().__init__()
        # 添加中文字体
        self.add_font("simhei", "", "C:/Windows/Fonts/simhei.ttf")
        self.add_font("simhei", "B", "C:/Windows/Fonts/simhei.ttf")

    def header(self):
        if self.page_no() > 1:
            self.set_font("simhei", "", 8)
            self.cell(0, 10, "stats-cli-py 技能架构文档", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
            self.line(10, 20, 200, 20)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("simhei", "", 8)
        self.cell(0, 10, f"第 {self.page_no()} 页", align="C")

    def chapter_title(self, title):
        self.set_font("simhei", "B", 16)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(5)

    def section_title(self, title):
        self.set_font("simhei", "B", 12)
        self.set_text_color(41, 128, 185)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font("simhei", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def code_block(self, text):
        self.set_font("simhei", "", 9)
        self.set_fill_color(240, 240, 240)
        self.multi_cell(0, 5, text, fill=True)
        self.ln(3)

    def table_row(self, cells, widths, bold=False):
        style = "B" if bold else ""
        self.set_font("simhei", style, 9)
        for i, (cell, w) in enumerate(zip(cells, widths)):
            if i % 2 == 0:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(255, 255, 255)
            self.cell(w, 7, str(cell), border=1, fill=True)
        self.ln()


def main():
    pdf = SkillPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # === 封面 ===
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("simhei", "B", 28)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 15, "stats-cli-py", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("simhei", "", 16)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "AI Agent 技能架构文档", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(10)
    pdf.set_font("simhei", "", 12)
    pdf.cell(0, 8, "构建方式 | 最佳实践 | AI Agent 集成", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(20)
    pdf.set_draw_color(41, 128, 185)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("simhei", "", 11)
    pdf.cell(0, 8, "版本: 1.2.1", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.cell(0, 8, "日期: 2026-06-15", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.cell(0, 8, "命令: 33 | 测试: 1033 | 覆盖率: 95%", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    # === 目录 ===
    pdf.add_page()
    pdf.chapter_title("目录")
    toc = [
        ("1. 技能概述", "stats-cli-py 是什么？"),
        ("2. 架构设计", "三层架构"),
        ("3. AI Agent 调用方式", "JSON 分发器模式"),
        ("4. 渐进式披露", "两层设计"),
        ("5. 最佳实践评分", "8 维度评估"),
        ("6. 决策树", "统计方法路由"),
        ("7. 自发现机制", "discover 命令"),
        ("8. 改进记录", "我们做了哪些改进"),
    ]
    for title, desc in toc:
        pdf.set_font("simhei", "B", 11)
        pdf.cell(80, 8, title)
        pdf.set_font("simhei", "", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)

    # === 第1章: 概述 ===
    pdf.add_page()
    pdf.chapter_title("1. 技能概述")
    pdf.body_text(
        "stats-cli-py 是一个纯 Python 统计分析 CLI/库，专为制造业和质量工程设计。"
        "它提供 33 个统计命令，涵盖描述性统计、假设检验、SPC 控制图、过程能力分析、"
        "DOE 实验设计、MSA 测量系统分析、可靠性分析、多变量分析、时间序列等领域。"
    )
    pdf.body_text("核心设计原则:")
    pdf.body_text("  - 所有 I/O 均为 JSON 格式: 通过 stdin/stdout 或文件交互")
    pdf.body_text("  - 纯 Python 实现: 基于 scipy + statsmodels，无 R 依赖")
    pdf.body_text("  - AI Agent 原生: 自描述 API，内置 discover 命令")
    pdf.body_text("  - 渐进式披露: 简单入口，按需深入")
    pdf.ln(5)
    pdf.section_title("命令分类 (33 个命令)")
    cats = [
        ("基础统计", "descriptive, normality, outlier"),
        ("假设检验", "ttest, anova, nonparametric, homogeneity, equivalence, power"),
        ("SPC 控制图", "control_chart, capability, trend"),
        ("回归分析", "regression, correlation"),
        ("实验设计", "doe (全因子, 分数因子, 田口, 响应曲面)"),
        ("测量系统", "gage_rr (交叉, 嵌套, 属性)"),
        ("可靠性", "reliability (威布尔, Kaplan-Meier)"),
        ("多变量", "multivariate (PCA, 聚类, 判别)"),
        ("时间序列", "timeseries (指数平滑, ARIMA, 分解)"),
        ("数据处理", "explore, clean, transform, discover"),
        ("工作流", "workflow, workflow_template, check_assumptions, recommend"),
        ("报告导出", "report, export_excel, export_pdf, run"),
    ]
    w = [40, 150]
    pdf.table_row(["分类", "命令"], w, bold=True)
    for cat, cmds in cats:
        pdf.table_row([cat, cmds], w)

    # === 第2章: 架构 ===
    pdf.add_page()
    pdf.chapter_title("2. 架构设计")
    pdf.body_text("技能采用三层架构:")
    pdf.ln(3)
    pdf.section_title("第1层: 路由层 (main.py)")
    pdf.body_text(
        "解析 JSON 输入，通过 _route() 函数路由命令（延迟导入），"
        "处理异常，将输出包装在标准信封中。共 374 行代码。"
    )
    pdf.ln(3)
    pdf.section_title("第2层: 执行层 (stats_engine/*.py)")
    pdf.body_text(
        "28 个独立模块，每个模块暴露一个公开函数，接受 **params 参数并返回 dict。"
        "所有计算使用 scipy.stats、numpy 和 statsmodels，无自定义统计实现。"
    )
    pdf.ln(3)
    pdf.section_title("第3层: 工具层 (utils/*.py)")
    pdf.body_text(
        "4 个工具模块: data_loader（文件加载，自动编码检测）、"
        "data_cleaner（NaN/Inf 处理）、validators（输入验证）、"
        "output（JSON 信封，数值四舍五入）。"
    )
    pdf.ln(5)
    pdf.section_title("数据流")
    pdf.code_block(
        "输入 JSON -> main.py handler() -> _route(command, params)\n"
        "    -> stats_engine/module.function(**params)\n"
        "    -> utils/output.success(data) 或 error(msg)\n"
        "    -> JSON 标准输出"
    )

    # === 第3章: AI Agent 调用 ===
    pdf.add_page()
    pdf.chapter_title("3. AI Agent 调用方式")
    pdf.section_title("三种调用模式")
    pdf.code_block(
        "# 模式 1: stdin 管道\n"
        "echo '{\"command\":\"descriptive\",\"params\":{\"values\":[1,2,3]}}' | python main.py\n\n"
        "# 模式 2: 文件参数\n"
        "python main.py input.json\n\n"
        "# 模式 3: Python 直接调用\n"
        "from main import handler\n"
        "result = handler({\"command\": \"descriptive\", \"params\": {\"values\": [1,2,3]}})"
    )
    pdf.ln(3)
    pdf.section_title("标准输出信封")
    pdf.code_block(
        "成功: {\"status\":\"success\", \"version\":\"1.2.1\", \"timestamp\":\"...\", \"data\":{...}}\n"
        "失败: {\"status\":\"error\", \"error_type\":\"...\", \"message\":\"...\", \"suggestion\":\"...\"}"
    )
    pdf.ln(3)
    pdf.section_title("Agent 调用流程 (7 步)")
    pdf.body_text("1. Agent 收到用户模糊请求 (\"帮我分析数据\")")
    pdf.body_text("2. Agent 调用 discover -> 获取 33 个命令摘要")
    pdf.body_text("3. Agent 根据 description/category 选择合适命令")
    pdf.body_text("4. Agent 调用 discover {command_name:\"xxx\"} -> 获取参数 schema")
    pdf.body_text("5. Agent 用 example 作为模板构造 JSON 请求")
    pdf.body_text("6. Agent 调用目标命令 -> 获取结果")
    pdf.body_text("7. Agent 用 output_fields 解析结果，向用户解释")

    # === 第4章: 渐进式披露 ===
    pdf.add_page()
    pdf.chapter_title("4. 渐进式披露")
    pdf.body_text("技能实现了两层渐进式披露:")
    pdf.ln(3)
    pdf.section_title("第1层: SKILL.md 静态文档 (4 级)")
    pdf.body_text("第1级: Quick Start 决策树 (5 个顶层分支)")
    pdf.body_text("  -> 第2级: 4 个详细决策树 (比较/关系/预测/质量控制)")
    pdf.body_text("    -> 第3级: 5 个场景化工作流 (完整 JSON 示例)")
    pdf.body_text("      -> 第4级: 33 个命令完整参考 (所有参数)")
    pdf.ln(3)
    pdf.section_title("第2层: discover 运行时动态发现 (2 级)")
    pdf.body_text("第1级: discover (无参数) -> 轻量摘要 (description + category)")
    pdf.body_text("  -> 第2级: discover {command_name:\"xxx\"} -> 完整 schema (params + output_fields)")
    pdf.ln(5)
    pdf.section_title("为什么需要两层?")
    pdf.body_text(
        "静态 SKILL.md 在技能激活时加载一次，提供决策树和场景覆盖常见用例。"
        "discover 命令在运行时查询，用于 SKILL.md 未覆盖的边缘场景，"
        "或当 Agent 需要精确参数 schema 时。"
    )

    # === 第5章: 最佳实践 ===
    pdf.add_page()
    pdf.chapter_title("5. 最佳实践评分")
    w = [45, 20, 125]
    pdf.table_row(["评估维度", "评分", "关键优势"], w, bold=True)
    scores = [
        ("触发条件", "Strong", "双语触发词，覆盖面广"),
        ("渐进式披露", "M-Strong", "两层设计，决策树入口"),
        ("决策引导", "Strong", "4 个决策树，recommend 命令"),
        ("错误处理", "Moderate", "7 种错误类型，suggestion 字段"),
        ("示例驱动", "Strong", "5 个场景，33 个命令示例"),
        ("自描述性", "Strong", "运行时 discover，结构化 params"),
        ("组合能力", "M-Strong", "workflow，9 个模板，auto_check"),
        ("参数文档", "Strong", "结构化 params，含 required/type"),
    ]
    for row in scores:
        pdf.table_row(row, w)
    pdf.ln(5)
    pdf.section_title("核心设计亮点")
    pdf.body_text("1. discover 命令: 运行时自描述，Agent 无需读文档即可了解能力")
    pdf.body_text("2. 决策树: 编码正确的统计方法论 (先检验正态性，再选参数/非参数)")
    pdf.body_text("3. recommend 命令: 元引导 - Agent 可以直接问工具该用什么方法")
    pdf.body_text("4. workflow_template: 9 个预制模板覆盖常见分析场景")
    pdf.body_text("5. 结构化参数: discover 中每个参数含 required/type/desc")

    # === 第6章: 决策树 ===
    pdf.add_page()
    pdf.chapter_title("6. 决策树")
    pdf.section_title("比较分析决策树")
    pdf.code_block(
        "用户想比较数据\n"
        "    |\n"
        "    +-- 只有 2 组?\n"
        "    |   +-- 正态? -> ttest (two_sample)\n"
        "    |   +-- 非正态? -> nonparametric mann_whitney\n"
        "    |       +-- 也考虑: transform boxcox -> 重新检验\n"
        "    |\n"
        "    +-- 3+ 组?\n"
        "    |   +-- 正态? -> anova one_way\n"
        "    |   +-- 非正态? -> nonparametric kruskal_wallis\n"
        "    |\n"
        "    +-- 配对数据?\n"
        "        +-- 正态? -> ttest paired\n"
        "        +-- 非正态? -> nonparametric wilcoxon"
    )
    pdf.ln(3)
    pdf.section_title("质量控制决策树")
    pdf.code_block(
        "用户想做质量控制\n"
        "    |\n"
        "    +-- 监控稳定性? -> control_chart (imr/xbar/r)\n"
        "    +-- 检查能力? -> capability (需要 USL/LSL)\n"
        "    +-- 测量系统? -> gage_rr (交叉/嵌套)\n"
        "    +-- 可靠性? -> reliability (威布尔/Kaplan-Meier)\n"
        "    +-- 检测趋势? -> trend (cusum/ewma/runs)"
    )

    # === 第7章: 自发现 ===
    pdf.add_page()
    pdf.chapter_title("7. 自发现机制 (discover 命令)")
    pdf.body_text(
        "discover 命令是运行时自描述机制，让 AI Agent 无需读取 SKILL.md "
        "即可了解可用命令。"
    )
    pdf.ln(3)
    pdf.section_title("三种查询模式")
    pdf.code_block(
        "# 模式 1: 列出所有命令 (轻量)\n"
        '{"command": "discover"}\n\n'
        "# 模式 2: 按类别筛选\n"
        '{"command": "discover", "params": {"category": "spc"}}\n\n'
        "# 模式 3: 单命令详情 (完整 schema)\n"
        '{"command": "discover", "params": {"command_name": "capability"}}'
    )
    pdf.ln(3)
    pdf.section_title("结构化参数 (v1.2.1)")
    pdf.body_text("每个命令现在暴露结构化参数元数据:")
    pdf.code_block(
        '"params": [\n'
        '  {"name": "values", "required": true, "type": "array", "desc": "过程数据"},\n'
        '  {"name": "usl",    "required": true, "type": "number", "desc": "规格上限"},\n'
        '  {"name": "lsl",    "required": true, "type": "number", "desc": "规格下限"},\n'
        '  {"name": "target", "required": false, "type": "number", "desc": "目标值"}\n'
        "]"
    )
    pdf.body_text("这消除了猜测 - Agent 准确知道哪些参数是必需的，哪些是可选的。")

    # === 第8章: 改进记录 ===
    pdf.add_page()
    pdf.chapter_title("8. 改进记录")
    pdf.section_title("v1.2.1 改动 (2026-06-15)")
    pdf.body_text("SKILL.md 改进:")
    pdf.body_text("  + 添加 Intent-to-Command 快速查找表 (19 个命令)")
    pdf.body_text("  + 在文档顶部推荐 Agent 优先使用 discover 命令")
    pdf.body_text("  + 添加负面触发词 (不支持: 流数据、文本、图像、非数值)")
    pdf.body_text("  + 添加 Top 5 命令输出示例")
    pdf.body_text("  + 添加错误恢复工作流表")
    pdf.body_text("  + 文档化 workflow auto_check 失败语义")
    pdf.body_text("  + 文档化 run 命令沙箱限制")
    pdf.body_text("  + 在决策树中添加 Box-Cox 交叉链接")
    pdf.body_text("  + 将所有占位 [...] 替换为可运行数据")
    pdf.ln(3)
    pdf.body_text("代码改进:")
    pdf.body_text("  + discover.py 添加结构化 params (33 个命令)")
    pdf.body_text("  + 修复 fpdf2 弃用警告 (report.py)")
    pdf.body_text("  + 修复 Matplotlib 弃用 (charts.py)")
    pdf.body_text("  + 异常改为 logging 记录而非吞没")
    pdf.ln(3)
    pdf.body_text("打包:")
    pdf.body_text("  + pyproject.toml: 版本 1.2.1，添加 fpdf2 依赖，修复 build-backend")
    pdf.body_text("  + Zip: 排除 TODO.md、CLAUDE.md、测试文件、构建产物")
    pdf.body_text("  + 最终 zip: 47 个文件，310 KB")

    # 保存
    pdf.output("D:/learn/claudecode/stats-cli/docs/skill-architecture-cn.pdf")
    print("PDF 已生成: docs/skill-architecture-cn.pdf")


if __name__ == "__main__":
    main()

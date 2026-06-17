"""生成演示用中文 PDF."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos


class PresentationPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("simhei", "", "C:/Windows/Fonts/simhei.ttf")
        self.add_font("simhei", "B", "C:/Windows/Fonts/simhei.ttf")

    def header(self):
        if self.page_no() > 1:
            self.set_font("simhei", "", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, "stats-cli-py 技能介绍", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
            self.set_text_color(0, 0, 0)
            self.line(10, 20, 200, 20)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("simhei", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"第 {self.page_no()} 页", align="C")
        self.set_text_color(0, 0, 0)

    def title_page(self, title, subtitle, author, department, date):
        self.add_page()
        self.ln(30)
        self.set_font("simhei", "B", 28)
        self.set_text_color(41, 128, 185)
        self.cell(0, 15, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(5)
        self.set_font("simhei", "", 16)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, subtitle, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(15)
        self.set_draw_color(41, 128, 185)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(15)
        self.set_text_color(0, 0, 0)
        self.set_font("simhei", "", 12)
        self.cell(0, 8, f"作者: {author}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.cell(0, 8, f"部门: {department}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.cell(0, 8, f"日期: {date}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    def section(self, title):
        self.ln(5)
        self.set_font("simhei", "B", 16)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(5)

    def subsection(self, title):
        self.set_font("simhei", "B", 13)
        self.set_text_color(52, 73, 94)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def text(self, t):
        self.set_font("simhei", "", 11)
        self.multi_cell(0, 7, t)
        self.ln(2)

    def bullet(self, t):
        self.set_font("simhei", "", 11)
        self.cell(10, 7, "")
        self.multi_cell(0, 7, f"  {t}")
        self.ln(1)

    def highlight(self, t):
        self.set_font("simhei", "", 11)
        self.set_fill_color(255, 249, 196)
        self.set_text_color(120, 80, 0)
        self.multi_cell(0, 7, f"  {t}", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def table(self, headers, rows, widths=None):
        if widths is None:
            widths = [int(190 / len(headers))] * len(headers)
        self.set_font("simhei", "B", 10)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(widths[i], 8, h, border=1, fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)
        self.set_font("simhei", "", 10)
        for idx, row in enumerate(rows):
            for i, cell in enumerate(row):
                if idx % 2 == 0:
                    self.set_fill_color(245, 245, 245)
                else:
                    self.set_fill_color(255, 255, 255)
                self.cell(widths[i], 7, str(cell), border=1, fill=True)
            self.ln()


def main():
    pdf = PresentationPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # === 封面 ===
    pdf.title_page(
        "stats-cli-py",
        "AI 驱动的统计分析技能",
        "吴思潭",
        "台州质量基地 - 验证部",
        "2026 年 6 月 15 日",
    )

    # === 第1章: 为什么需要这个 Skill ===
    pdf.add_page()
    pdf.section("一、为什么需要这个 Skill?")

    pdf.subsection("现实问题")
    pdf.text("在日常质量工作中，我们经常需要进行统计分析：")
    pdf.bullet("比较两批产品的均值是否有显著差异")
    pdf.bullet("评估生产过程的能力是否满足规格要求")
    pdf.bullet("监控过程是否稳定，有没有异常波动")
    pdf.bullet("分析测量系统的重复性和再现性")
    pdf.ln(3)

    pdf.subsection("当前痛点")
    pdf.highlight("JMP 不是每台电脑都有——许可证昂贵，安装受限")
    pdf.highlight("Excel 功能有限——复杂分析需要手动操作，容易出错")
    pdf.highlight("数据分析需求随时产生——会议室、现场、出差途中")
    pdf.highlight("等待专业人员分析——响应慢，沟通成本高")
    pdf.ln(3)

    pdf.subsection("AI 时代的新可能")
    pdf.text("AI Agent（如 Claude、ChatGPT）擅长理解自然语言和结构化推理。")
    pdf.text("如果我们把统计分析能力封装成一个「技能」，让 AI Agent 调用，就能实现：")
    pdf.bullet("随时随地分析——只需要一个 AI 对话窗口")
    pdf.bullet("零代码操作——用自然语言描述需求即可")
    pdf.bullet("即时响应——几秒钟完成分析，不需要等待")
    pdf.bullet("专业可靠——基于 scipy/statsmodels，结果可追溯")

    # === 第2章: 解决了什么问题 ===
    pdf.add_page()
    pdf.section("二、解决了什么问题?")

    pdf.subsection("核心价值")
    pdf.text("stats-cli-py 让「人人可用统计分析」成为可能：")
    pdf.ln(3)

    w = [60, 130]
    pdf.table(
        ["之前", "之后"],
        [
            ["需要 JMP/SPSS 许可证", "免费，开源"],
            ["需要安装软件", "AI 对话窗口直接用"],
            ["需要学习软件操作", "用自然语言描述需求"],
            ["分析需要 30 分钟+", "几秒钟出结果"],
            ["需要专业统计知识", "AI 自动推荐方法"],
            ["结果需要手动整理", "自动生成 JSON/PDF/Excel"],
        ],
        w,
    )
    pdf.ln(5)

    pdf.subsection("覆盖的分析场景")
    pdf.text("33 个统计命令，覆盖质量工程 90% 的常见分析需求：")
    pdf.ln(2)
    pdf.table(
        ["类别", "功能"],
        [
            ["基础统计", "描述性统计、正态性检验、异常值检测"],
            ["假设检验", "t 检验、ANOVA、非参数检验、等价检验"],
            ["SPC 控制图", "I-MR、Xbar-R、p/np/c/u、EWMA、CUSUM"],
            ["过程能力", "Cp/Cpk/Pp/Ppk、Box-Cox 变换"],
            ["测量系统", "Gage R&R（交叉/嵌套/属性）"],
            ["实验设计", "全因子/分数因子/田口/响应曲面"],
            ["可靠性", "威布尔分析、Kaplan-Meier、寿命分布"],
            ["回归分析", "线性/多项式/逻辑/非线性（10 种）"],
            ["多变量", "PCA、聚类、判别分析"],
            ["时间序列", "指数平滑、ARIMA、分解"],
        ],
        w,
    )

    # === 第3章: 技术架构 ===
    pdf.add_page()
    pdf.section("三、技术架构")

    pdf.subsection("三层架构")
    pdf.text("stats-cli-py 采用清晰的三层架构设计：")
    pdf.ln(3)

    pdf.set_font("simhei", "", 11)
    pdf.set_fill_color(240, 240, 240)
    pdf.multi_cell(
        0,
        7,
        (
            "┌─────────────────────────────────────────────┐\n"
            "│              路由层 (main.py)                │\n"
            "│   JSON 输入 → 命令分发 → JSON 输出          │\n"
            "├─────────────────────────────────────────────┤\n"
            "│           执行层 (stats_engine/)             │\n"
            "│   28 个独立模块，每个命令一个文件            │\n"
            "├─────────────────────────────────────────────┤\n"
            "│            工具层 (utils/)                   │\n"
            "│   数据加载、清洗、验证、输出格式化           │\n"
            "└─────────────────────────────────────────────┘"
        ),
        fill=True,
    )
    pdf.ln(5)

    pdf.subsection("关键技术特点")
    pdf.bullet("纯 Python 实现——基于 scipy + statsmodels，无 R 依赖")
    pdf.bullet("JSON 输入输出——标准化接口，易于集成")
    pdf.bullet("延迟导入——只加载需要的模块，启动速度快")
    pdf.bullet("自描述 API——discover 命令可查询所有可用功能")
    pdf.bullet("图表生成——22 个命令支持自动生成图表（base64 PNG）")
    pdf.ln(3)

    pdf.subsection("代码质量")
    pdf.table(
        ["指标", "值"],
        [
            ["测试覆盖率", "95%"],
            ["测试总数", "964"],
            ["代码行数", "~3400 行"],
            ["模块数", "28 个统计模块 + 4 个工具模块"],
            ["CI/CD", "GitHub Actions（6 矩阵测试）"],
        ],
    )

    # === 第4章: 渐进式披露 ===
    pdf.add_page()
    pdf.section("四、渐进式披露设计")

    pdf.subsection("什么是渐进式披露?")
    pdf.text("渐进式披露是一种设计理念：先展示简单入口，用户需要时再提供更深层的细节。")
    pdf.text("就像手机的「设置」——常用功能在首页，高级选项藏在子菜单里。")
    pdf.ln(3)

    pdf.subsection("stats-cli-py 的两层渐进式披露")
    pdf.ln(2)

    pdf.set_font("simhei", "B", 11)
    pdf.cell(0, 8, "第 1 层: SKILL.md 静态文档（4 级）", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("simhei", "", 11)
    pdf.ln(2)
    pdf.set_fill_color(240, 240, 240)
    pdf.multi_cell(
        0,
        7,
        (
            "第 1 级: Quick Start 决策树（5 个顶层分支）\n"
            "    ↓ 需要更多细节时\n"
            "第 2 级: 4 个详细决策树（比较/关系/预测/质量控制）\n"
            "    ↓ 需要完整示例时\n"
            "第 3 级: 5 个场景化工作流（含完整 JSON 示例）\n"
            "    ↓ 需要查所有命令时\n"
            "第 4 级: 33 个命令完整参考"
        ),
        fill=True,
    )
    pdf.ln(5)

    pdf.set_font("simhei", "B", 11)
    pdf.cell(0, 8, "第 2 层: discover 运行时动态发现（2 级）", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("simhei", "", 11)
    pdf.ln(2)
    pdf.set_fill_color(240, 240, 240)
    pdf.multi_cell(
        0,
        7,
        (
            "第 1 级: discover（无参数）→ 返回 33 个命令的简要描述\n"
            "    ↓ 找到目标命令后\n"
            '第 2 级: discover {command_name:"xxx"} → 返回完整参数 schema'
        ),
        fill=True,
    )
    pdf.ln(5)

    pdf.subsection("为什么这样设计?")
    pdf.bullet("AI Agent 不需要一次加载全部信息——节省 token")
    pdf.bullet("常见场景快速路由——决策树直接给出推荐命令")
    pdf.bullet("边缘场景按需查询——discover 命令动态获取参数")
    pdf.bullet("新命令自动可发现——无需更新文档即可被 Agent 使用")

    # === 第5章: 如何使用 ===
    pdf.add_page()
    pdf.section("五、如何使用?")

    pdf.subsection("你只需要做一件事")
    pdf.highlight("告诉 AI 助手你想分析什么，它会自动完成。")
    pdf.ln(3)

    pdf.subsection("使用示例")
    pdf.ln(2)

    pdf.set_font("simhei", "B", 11)
    pdf.cell(0, 8, "示例 1: 查看数据基本情况", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("simhei", "", 11)
    pdf.text("你说: 「帮我看看这批数据的基本情况」")
    pdf.text("AI 助手: 自动调用 descriptive，返回均值、标准差、置信区间等")
    pdf.ln(2)

    pdf.set_font("simhei", "B", 11)
    pdf.cell(0, 8, "示例 2: 比较两批产品", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("simhei", "", 11)
    pdf.text("你说: 「A 批和 B 批产品的均值有没有显著差异?」")
    pdf.text("AI 助手: 先检查正态性，再选择 t 检验或非参数检验，返回 p 值和结论")
    pdf.ln(2)

    pdf.set_font("simhei", "B", 11)
    pdf.cell(0, 8, "示例 3: 评估过程能力", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("simhei", "", 11)
    pdf.text("你说: 「这个过程的能力怎么样? 规格是 9 到 11」")
    pdf.text("AI 助手: 自动调用 capability，返回 Cp/Cpk 和能力等级")
    pdf.ln(2)

    pdf.set_font("simhei", "B", 11)
    pdf.cell(0, 8, "示例 4: 不知道用什么方法?", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("simhei", "", 11)
    pdf.text("你说: 「我想分析这个数据，但不知道用什么方法」")
    pdf.text("AI 助手: 调用 recommend 命令，根据你的目标推荐最合适的分析方法")
    pdf.ln(3)

    pdf.subsection("支持的输入方式")
    pdf.bullet("直接输入数据: 「数据是 10.1, 10.2, 10.3, ...」")
    pdf.bullet("上传文件: Excel、CSV、JSON、文本文件均可")
    pdf.bullet("描述需求: 「帮我比较两批产品」「看看过程能力」")
    pdf.ln(3)

    pdf.subsection("输出格式")
    pdf.bullet("JSON 结构化数据——可被程序处理")
    pdf.bullet("自动生成图表——直方图、控制图、Q-Q 图等")
    pdf.bullet("可导出 Excel/PDF 报告")

    # === 第6章: 开发工具 ===
    pdf.add_page()
    pdf.section("六、开发工具")

    pdf.subsection("使用 Claude Code 开发")
    pdf.text("stats-cli-py 使用 Claude Code（Anthropic 官方 CLI 工具）进行开发。")
    pdf.text("Claude Code 是一个 AI 编程助手，可以直接在终端中帮助编写、调试和优化代码。")
    pdf.ln(3)

    pdf.subsection("开发过程")
    pdf.bullet("需求分析: 与 Claude Code 对话，明确功能需求")
    pdf.bullet("代码实现: Claude Code 生成代码，人工审查确认")
    pdf.bullet("测试验证: 自动运行测试，确保代码正确")
    pdf.bullet("迭代优化: 根据测试结果持续改进")
    pdf.ln(3)

    pdf.subsection("开发效率")
    pdf.text("使用 Claude Code 开发的效率提升：")
    pdf.bullet("33 个统计命令——从零到完成约 2 周")
    pdf.bullet("964 个测试——覆盖率 95%")
    pdf.bullet("完整的 SKILL.md 文档——AI Agent 可直接使用")
    pdf.bullet("生产就绪的 zip 包——可直接分发")
    pdf.ln(3)

    pdf.highlight("Claude Code 让「一个人 + AI」就能完成传统上需要团队才能完成的工作。")

    # === 第7章: 总结 ===
    pdf.add_page()
    pdf.section("七、总结")

    pdf.subsection("stats-cli-py 的价值")
    pdf.ln(2)
    pdf.table(
        ["维度", "价值"],
        [
            ["可及性", "人人可用，无需专业软件和许可证"],
            ["效率", "几秒钟完成传统 30 分钟的分析"],
            ["专业性", "基于 scipy/statsmodels，结果可追溯"],
            ["覆盖面", "33 个命令，覆盖 90% 质量分析场景"],
            ["可靠性", "95% 测试覆盖率，964 个测试"],
            ["可扩展", "模块化架构，易于添加新功能"],
        ],
    )
    pdf.ln(5)

    pdf.subsection("未来展望")
    pdf.bullet("MCP Server 支持——让 Claude Desktop 直接调用")
    pdf.bullet("更多图表类型——残差图、ACF 图、功效曲线等")
    pdf.bullet("REST API——支持 Web 应用集成")
    pdf.bullet("更多语言支持——英文界面，国际化部署")
    pdf.ln(5)

    pdf.subsection("致谢")
    pdf.text("感谢 Claude Code 在开发过程中提供的强大支持。")
    pdf.text("感谢台州质量基地验证部的支持。")
    pdf.ln(5)

    pdf.set_font("simhei", "B", 14)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 10, "谢谢!", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    # 保存
    pdf.output("D:/learn/claudecode/stats-cli/docs/presentation-cn.pdf")
    print("PDF 已生成: docs/presentation-cn.pdf")


if __name__ == "__main__":
    main()

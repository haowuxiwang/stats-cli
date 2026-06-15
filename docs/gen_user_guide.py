"""生成面向非技术人员的 skill 使用指南 PDF."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos


class UserGuidePDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("simhei", "", "C:/Windows/Fonts/simhei.ttf")
        self.add_font("simhei", "B", "C:/Windows/Fonts/simhei.ttf")

    def header(self):
        if self.page_no() > 1:
            self.set_font("simhei", "", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, "stats-cli-py Skill 使用指南", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
            self.set_text_color(0, 0, 0)
            self.line(10, 20, 200, 20)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("simhei", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"第 {self.page_no()} 页", align="C")
        self.set_text_color(0, 0, 0)

    def big_title(self, text):
        self.set_font("simhei", "B", 22)
        self.set_text_color(41, 128, 185)
        self.cell(0, 15, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_text_color(0, 0, 0)
        self.ln(5)

    def chapter(self, num, title):
        self.ln(3)
        self.set_font("simhei", "B", 16)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, f"  {num}. {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(5)

    def subtitle(self, text):
        self.set_font("simhei", "B", 12)
        self.set_text_color(52, 73, 94)
        self.cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def text(self, t):
        self.set_font("simhei", "", 11)
        self.multi_cell(0, 7, t)
        self.ln(2)

    def cmd(self, name, desc):
        self.set_font("simhei", "B", 11)
        self.set_text_color(41, 128, 185)
        self.cell(45, 7, name)
        self.set_text_color(0, 0, 0)
        self.set_font("simhei", "", 11)
        self.cell(0, 7, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def tip(self, t):
        self.set_font("simhei", "", 10)
        self.set_fill_color(255, 249, 196)
        self.set_text_color(120, 80, 0)
        self.multi_cell(0, 6, f"  {t}", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def bullet(self, t):
        self.set_font("simhei", "", 11)
        self.cell(10, 7, "")
        self.multi_cell(0, 7, f"  {t}")
        self.ln(1)

    def table(self, headers, rows):
        w = [int(190 / len(headers))] * len(headers)
        self.set_font("simhei", "B", 10)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(w[i], 8, h, border=1, fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)
        self.set_font("simhei", "", 10)
        for idx, row in enumerate(rows):
            for i, cell in enumerate(row):
                if idx % 2 == 0:
                    self.set_fill_color(245, 245, 245)
                else:
                    self.set_fill_color(255, 255, 255)
                self.cell(w[i], 7, str(cell), border=1, fill=True)
            self.ln()


def main():
    pdf = UserGuidePDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # === 封面 ===
    pdf.add_page()
    pdf.ln(30)
    pdf.big_title("stats-cli-py")
    pdf.ln(5)
    pdf.set_font("simhei", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "统计分析 Skill 使用指南", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(5)
    pdf.set_font("simhei", "", 12)
    pdf.cell(0, 8, "面向质量工程师、生产管理人员、数据分析新手", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(20)
    pdf.set_draw_color(41, 128, 185)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(15)
    pdf.set_font("simhei", "", 11)
    pdf.cell(0, 8, "版本 1.2.1 | 2026 年 6 月", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    # === 第1章: 这是什么 ===
    pdf.add_page()
    pdf.chapter(1, "这是什么?")
    pdf.text(
        "stats-cli-py 是一个「统计分析 Skill」——它可以被 AI 助手（AI Agent）调用，"
        "帮你完成各种数据分析任务。"
    )
    pdf.ln(2)
    pdf.subtitle("什么是 Skill?")
    pdf.text(
        "Skill 就像 AI 助手的「技能包」。就像一个人学会了 Excel 就能做表格一样，"
        "AI 助手加载了这个 Skill，就能做统计分析。"
    )
    pdf.text("你不需要自己操作这个工具——你只需要告诉 AI 助手你想分析什么，它会自动调用 Skill 来完成。")
    pdf.ln(3)

    pdf.subtitle("它能帮你做什么?")
    pdf.ln(2)

    table_data = [
        ["看数据长什么样", "用 explore 查看数据结构、列名、缺失值"],
        ["检查数据是否正常", "用 normality 做正态性检验"],
        ["比较两批产品", "用 ttest 或 nonparametric 判断差异是否显著"],
        ["评估过程能力", "用 capability 计算 Cp/Cpk"],
        ["画控制图", "用 control_chart 监控过程稳定性"],
        ["找异常值", "用 outlier 自动检测异常点"],
        ["做回归分析", "用 regression 分析变量关系"],
        ["设计实验", "用 doe 生成实验方案"],
        ["分析测量系统", "用 gage_rr 做 Gage R&R 分析"],
        ["生成报告", "用 report 汇总结果，export_excel/export_pdf 导出"],
    ]
    pdf.table(["你想做什么", "对应命令"], table_data)

    pdf.ln(5)
    pdf.tip("你不需要记住这些命令——AI 助手会根据你的描述自动选择合适的方法。")

    # === 第2章: 怎么用 ===
    pdf.add_page()
    pdf.chapter(2, "怎么使用它?")
    pdf.text("使用这个 Skill 就像和同事对话一样——你告诉 AI 助手你想做什么，它会自动完成。")
    pdf.ln(3)

    pdf.subtitle("你只需要用自然语言描述需求")
    pdf.text("举几个例子:")
    pdf.ln(2)
    pdf.text("  你说: 「帮我看看这批数据的基本情况」")
    pdf.text("  AI 助手会自动调用 descriptive 命令，告诉你平均值、标准差等")
    pdf.ln(2)
    pdf.text("  你说: 「检验一下这批数据是否符合正态分布」")
    pdf.text("  AI 助手会自动调用 normality 命令，告诉你检验结果")
    pdf.ln(2)
    pdf.text("  你说: 「比较一下 A 批和 B 批产品有没有显著差异」")
    pdf.text("  AI 助手会先检查正态性，再选择 ttest 或 nonparametric")
    pdf.ln(3)

    pdf.subtitle("不知道该用什么方法?")
    pdf.text("如果你不确定该用哪种分析方法，直接告诉 AI 助手你的目标:")
    pdf.bullet("「我想比较两批产品的均值有没有差异」")
    pdf.bullet("「我想看看这个过程能不能满足规格要求」")
    pdf.bullet("「我想分析温度和产量之间的关系」")
    pdf.text("AI 助手会自动选择最合适的命令和参数。")
    pdf.ln(3)
    pdf.tip("AI 助手内置了「推荐」功能——它可以问你几个问题，然后建议最合适的分析方法。")

    # === 第3章: 常见场景 ===
    pdf.add_page()
    pdf.chapter(3, "常见使用场景")
    pdf.ln(3)

    pdf.subtitle("场景 1: 我有一批数据，想看看基本情况")
    pdf.text("AI 助手会调用 descriptive 命令")
    pdf.text("结果包含: 数据个数、平均值、中位数、标准差、最大最小值、四分位数、95% 置信区间")
    pdf.text("就像体检报告里的「基本信息」——先看看总体情况。")
    pdf.ln(3)

    pdf.subtitle("场景 2: 我想知道数据是否符合正态分布")
    pdf.text("AI 助手会调用 normality 命令")
    pdf.text("结果包含: 三种检验方法（Shapiro-Wilk、Anderson-Darling、Lilliefors）的结果，以及综合判断")
    pdf.text("为什么要检查? 因为很多统计方法（如 t 检验）要求数据是正态分布的。")
    pdf.ln(3)

    pdf.subtitle("场景 3: 我想比较两批产品有没有差异")
    pdf.text("AI 助手会根据数据特征自动选择:")
    pdf.bullet("正态数据 → 调用 ttest（参数检验）")
    pdf.bullet("非正态数据 → 调用 nonparametric（非参数检验）")
    pdf.text("结果包含: p 值（< 0.05 表示有显著差异）和置信区间")
    pdf.ln(3)

    pdf.subtitle("场景 4: 我想评估生产过程能力")
    pdf.text("AI 助手会调用 capability 命令")
    pdf.text("需要提供: 数据 + 规格上限 (USL) + 规格下限 (LSL)")
    pdf.text("结果包含: Cp、Cpk、Pp、Ppk 等指标，以及过程能力等级")
    pdf.text("这是质量工程师最常用的工具——判断生产过程是否满足客户要求。")

    # === 第4章: 更多场景 ===
    pdf.add_page()
    pdf.subtitle("场景 5: 我想监控生产过程是否稳定")
    pdf.text("AI 助手会调用 control_chart 命令")
    pdf.text("支持的图表: I-MR（个体值）、Xbar-R（子组）、p/np/c/u（计数型）、EWMA、CUSUM")
    pdf.text("结果包含: 控制限、中心线、异常点、是否违反判异规则")
    pdf.ln(3)

    pdf.subtitle("场景 6: 我想分析两个变量的关系")
    pdf.text("AI 助手会根据你的目标选择:")
    pdf.bullet("相关分析 → 调用 correlation（分析相关程度）")
    pdf.bullet("回归分析 → 调用 regression（建立数学模型）")
    pdf.text("支持 10 种回归类型: 线性、多项式、逻辑回归、指数、幂函数等")
    pdf.ln(3)

    pdf.subtitle("场景 7: 我想设计一个实验")
    pdf.text("AI 助手会调用 doe 命令")
    pdf.text("支持: 全因子设计、分数因子设计、田口设计、响应曲面设计")
    pdf.text("它会帮你生成实验方案，告诉你需要做多少次实验")
    pdf.ln(3)

    pdf.subtitle("场景 8: 我想分析测量系统的可靠性")
    pdf.text("AI 助手会调用 gage_rr 命令")
    pdf.text("支持: 交叉设计、嵌套设计、属性分析")
    pdf.text("结果包含: 重复性、再现性、ndc（可区分的类别数）、评级")

    # === 第5章: 最佳实践 ===
    pdf.add_page()
    pdf.chapter(4, "什么是「最佳实践」?")
    pdf.text(
        "「最佳实践」就是经过验证的、最有效的工作方式。"
        "就像质量工程里有「先检验再判断」的原则一样，"
        "数据分析也有一些公认的好习惯。"
    )
    pdf.ln(3)

    pdf.subtitle("最佳实践 1: 先看数据，再分析")
    pdf.text("在做任何分析之前，先用 explore 看看数据:")
    pdf.bullet("数据有多少行、多少列?")
    pdf.bullet("有没有缺失值（空白）?")
    pdf.bullet("数据类型对不对（是不是数字）?")
    pdf.text("就像医生看病先问诊一样——先了解基本情况，再做检查。")
    pdf.ln(3)

    pdf.subtitle("最佳实践 2: 先检验正态性，再选方法")
    pdf.text("很多统计方法（如 ttest、anova）要求数据符合正态分布。")
    pdf.text("正确的流程:")
    pdf.bullet("第 1 步: 用 normality 检查数据是否正态")
    pdf.bullet("第 2 步: 如果正态 → 用 ttest、anova 等参数检验")
    pdf.bullet("第 3 步: 如果非正态 → 用 nonparametric 等非参数检验")
    pdf.bullet("也可以: 先用 transform（Box-Cox 变换）把数据变成正态，再用参数检验")
    pdf.ln(3)

    pdf.subtitle("最佳实践 3: 检查测量系统是否可靠")
    pdf.text("在分析数据之前，确保你的测量系统是可靠的:")
    pdf.bullet("用 gage_rr 评估测量设备的重复性和再现性")
    pdf.bullet("如果测量系统本身误差太大，分析结果也不可信")
    pdf.text("就像用不准的秤称重——秤本身有问题，称出来的数据就没意义。")

    # === 第6章: 更多最佳实践 ===
    pdf.add_page()
    pdf.subtitle("最佳实践 4: 关注规格限和目标值")
    pdf.text("做过程能力分析时，一定要提供:")
    pdf.bullet("规格上限 (USL): 客户允许的最大值")
    pdf.bullet("规格下限 (LSL): 客户允许的最小值")
    pdf.bullet("目标值 (Target): 理想的中心值（可选）")
    pdf.text("没有规格限，就无法判断过程是否「合格」——就像考试不知道及格线一样。")
    pdf.ln(3)

    pdf.subtitle("最佳实践 5: 用控制图持续监控")
    pdf.text("control_chart 就像「过程的仪表盘」:")
    pdf.bullet("中心线: 过程的平均值")
    pdf.bullet("控制限: 过程的正常波动范围（通常是 3 西格玛）")
    pdf.bullet("异常点: 超出控制限的点，说明过程可能出了问题")
    pdf.text("建议: 关键工序定期画控制图，及时发现异常。")
    pdf.ln(3)

    pdf.subtitle("最佳实践 6: 用工作流自动化重复分析")
    pdf.text("如果你经常做同样的分析流程（比如每批产品都做「描述统计 + 正态性 + 能力分析」），")
    pdf.text("可以用 workflow 命令把多个步骤串起来，一次运行全部完成。")
    pdf.text("也有预制模板: workflow_template，比如 manufacturing 模板包含完整的制造分析流程。")
    pdf.ln(3)

    pdf.subtitle("最佳实践 7: 用 discover 了解可用功能")
    pdf.text("如果你想知道这个 Skill 还有哪些功能，可以让 AI 助手调用 discover 命令:")
    pdf.bullet("discover: 列出所有可用命令")
    pdf.bullet("discover {category: \"spc\"}: 只看 SPC 相关的命令")
    pdf.bullet("discover {command_name: \"capability\"}: 查看某个命令的详细参数")
    pdf.text("每个命令都会告诉你: 需要哪些参数（哪些是必需的）、返回什么结果。")

    # === 第7章: 总结 ===
    pdf.add_page()
    pdf.chapter(5, "总结")
    pdf.ln(3)
    pdf.text("stats-cli-py 是一个强大的统计分析 Skill，配合 AI 助手使用，它可以帮你:")
    pdf.bullet("快速了解数据的基本特征（descriptive）")
    pdf.bullet("选择正确的统计分析方法（normality → ttest/nonparametric）")
    pdf.bullet("评估过程能力是否满足要求（capability）")
    pdf.bullet("监控生产过程是否稳定（control_chart）")
    pdf.bullet("发现数据中的异常和趋势（outlier, trend）")
    pdf.bullet("自动生成分析报告（report, export_excel, export_pdf）")
    pdf.ln(5)
    pdf.text("记住三个核心原则:")
    pdf.bullet("先看数据，再分析（explore → normality → 分析命令）")
    pdf.bullet("先检验正态性，再选方法（正态 → ttest，非正态 → nonparametric）")
    pdf.bullet("先检查测量系统，再下结论（gage_rr → capability）")
    pdf.ln(5)
    pdf.tip("遇到问题? 告诉 AI 助手你的目标，它会用 recommend 命令建议合适的分析方法。")

    # 保存
    pdf.output("D:/learn/claudecode/stats-cli/docs/user-guide-cn-v2.pdf")
    print("PDF 已生成: docs/user-guide-cn-v2.pdf")


if __name__ == "__main__":
    main()

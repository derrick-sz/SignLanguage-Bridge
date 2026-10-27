"""Round-3 评审表 XLSX 生成（Rubric v2，双评审 A/B 各一份）。

被 make_round3_package.py 调用：读 CSV → 写带下拉校验、冻结窗格与说明页的 xlsx。
运行环境：C:\\Python314\\python.exe（openpyxl）；.conda-env 无 openpyxl 时 build 会跳过 xlsx。

Rubric v2 相对 v1 的关键变化：
- 删除单一「可用于交流」列，改为两列：
  「可理解部分信息」（能猜到大意）与「可支持可靠沟通」（含义准确到可直接使用）
- 「严重语义错误」给出与评分正交的定义（六槽位中任意一项被改变且会导致误解）
"""

from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

YESNO_UNCERTAIN = ("是", "否", "无法判断")
FILLABLE = {
    "评分0-4": ("0", "1", "2", "3", "4"),
    "人物_是否一致": YESNO_UNCERTAIN,
    "否定_是否一致": YESNO_UNCERTAIN,
    "数字_是否一致": YESNO_UNCERTAIN,
    "时间_是否一致": YESNO_UNCERTAIN,
    "动作_是否一致": YESNO_UNCERTAIN,
    "对象_是否一致": YESNO_UNCERTAIN,
    "严重语义错误": ("是", "否"),
    "可理解部分信息": YESNO_UNCERTAIN,
    "可支持可靠沟通": YESNO_UNCERTAIN,
    "流畅度1-5": ("1", "2", "3", "4", "5"),
}
TEXT_FILLABLE = ("备注", "评审人", "评审日期")
WIDTHS = {
    "行号": 6, "review_id": 10, "参考译文_EVAL_ONLY_仅评审用": 40, "候选译文": 40,
    "评分0-4": 8, "人物_是否一致": 12, "否定_是否一致": 12, "数字_是否一致": 12,
    "时间_是否一致": 12, "动作_是否一致": 12, "对象_是否一致": 12,
    "严重语义错误": 12, "可理解部分信息": 14, "可支持可靠沟通": 15, "流畅度1-5": 10,
    "备注": 30, "评审人": 12, "评审日期": 12,
}

INSTRUCTIONS = [
    ("Round-3 人工语义评审（Rubric v2）· 填写说明", None),
    ("评审对象", "三条候选译文（模型身份已隐藏、全局盲序、同一样本候选不相邻）"),
    ("评审人", "本轮为双人独立盲评：A/B 两份表由两位评审各自独立填写，互不参考"),
    ("填写方式", "只填黄色列；灰色列（行号/review_id/参考译文/候选译文）只读"),
    ("保存", "各自另存为 round3_reviewer_A_filled.xlsx / round3_reviewer_B_filled.xlsx"),
    ("汇总", "C:\\Python314\\python.exe make_round3_package.py stats --reviewer-a <A> --reviewer-b <B>"),
    ("评分0-4", "0 完全错误/相反；1 主要错误；2 部分正确但关键缺漏；3 核心意思正确、局部偏差；4 完整忠实自然"),
    ("严重语义错误（v2 正交定义）",
     "人物/否定/数字/时间/动作/对象 六项中**任意一项被改变**，且该改变会让读者得到与参考不同的信息 → 是。"
     "它衡量“事实是否被改变”，与整体评分相互独立（评分可为 3 但仍判“是”）"),
    ("可理解部分信息", "不看参考，这句话能否让人**大致猜到大意**（哪怕有明显错误）→ 是/否/无法判断"),
    ("可支持可靠沟通", "**本轮新增、最关键**：这句话的含义是否准确到可以**直接用于交流**（不需要听者再猜或纠正）→ 是/否/无法判断。"
                       "判定建议：仅当 评分≥3 且 严重语义错误=否 时才填“是”"),
    ("流畅度1-5", "1 不成句 … 5 完全自然"),
    ("六槽位", "与参考是否一致：人物/否定/数字/时间/动作/对象（是/否/无法判断）"),
    ("仲裁", "两人在 评分 或 严重语义错误 上不一致的行，由第三人或用户裁定（记录在汇总报告）"),
    ("边界", "参考译文仅供判定语义，不得回灌模型/提示词；数据为 CSL-Daily 1086 子集，非新盲测"),
]

SCALE = [
    ("评分量表（0-4 语义忠实度）", ""),
    ("0", "完全错误 / 核心意思相反"),
    ("1", "主要错误：关键信息明显错误"),
    ("2", "部分正确：能看出部分意思，但关键信息缺漏或明显偏差"),
    ("3", "核心意思正确：局部表达/细节偏差，不影响可靠理解"),
    ("4", "语义完整忠实且表达自然"),
    ("严重语义错误（正交）", "六槽位任一项被改变且会导致误解 → 是（与评分独立，评分 3 也可能是）"),
    ("可理解部分信息", "能大致猜到大意即可 → 是"),
    ("可支持可靠沟通", "含义准确到可直接用于交流（建议 评分≥3 且 严重错误=否）→ 是"),
    ("流畅度1-5", "1 不成句 … 5 完全自然"),
]


def _sheet_instructions(ws) -> None:
    ws.append(["项", "说明"])
    ws["A1"].font = Font(bold=True)
    ws["B1"].font = Font(bold=True)
    for key, value in INSTRUCTIONS:
        ws.append([key, value])
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 115
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")


def _sheet_scale(ws) -> None:
    for row in SCALE:
        ws.append(list(row))
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 105


def write_workbook(csv_path: Path, xlsx_path: Path, columns: list[str], *, reviewer: str) -> dict:
    with open(csv_path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    wb = Workbook()
    sheet = wb.active
    sheet.title = "评审表"
    sheet.append(columns)
    head_fill = PatternFill("solid", fgColor="D9D9D9")
    fill_fill = PatternFill("solid", fgColor="FFF2CC")
    for index, name in enumerate(columns, 1):
        cell = sheet.cell(row=1, column=index)
        cell.font = Font(bold=True)
        cell.fill = fill_fill if name in FILLABLE or name in TEXT_FILLABLE else head_fill
        sheet.column_dimensions[get_column_letter(index)].width = WIDTHS.get(name, 14)
    for row in rows:
        sheet.append([row.get(name, "") for name in columns])
    sheet.freeze_panes = "E2"

    validations = 0
    for index, name in enumerate(columns, 1):
        choices = FILLABLE.get(name)
        if not choices:
            continue
        letter = get_column_letter(index)
        dv = DataValidation(type="list", formula1='"' + ",".join(choices) + '"', allow_blank=True)
        dv.add(f"{letter}2:{letter}{len(rows) + 1}")
        sheet.add_data_validation(dv)
        validations += 1

    _sheet_instructions(wb.create_sheet("填写说明"))
    _sheet_scale(wb.create_sheet("评分量表"))
    progress = wb.create_sheet("进度统计")
    progress["A1"] = "评审人（本文件）"
    progress["B1"] = reviewer
    progress["A2"] = "已填写行数（按 评分 计）"
    col = get_column_letter(columns.index("评分0-4") + 1)
    progress["B2"] = f"=COUNT('评审表'!{col}2:{col}{len(rows) + 1})"
    progress["A3"] = "总行数"
    progress["B3"] = len(rows)
    progress.column_dimensions["A"].width = 24
    progress.column_dimensions["B"].width = 14

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)
    return {"rows": len(rows), "columns": len(columns), "validations": validations, "reviewer": reviewer}


if __name__ == "__main__":  # pragma: no cover
    import sys
    csv_in, xlsx_out = Path(sys.argv[1]), Path(sys.argv[2])
    reviewer = sys.argv[3] if len(sys.argv) > 3 else "?"
    with open(csv_in, encoding="utf-8-sig") as fh:
        cols = next(csv.reader(fh))
    print(write_workbook(csv_in, xlsx_out, cols, reviewer=reviewer))
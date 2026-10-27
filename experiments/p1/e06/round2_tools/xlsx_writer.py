"""Round-2 评审表 XLSX 生成（openpyxl）。

被 make_round2_package.py 调用：读 CSV → 写带下拉校验、冻结窗格与说明页的 xlsx。
运行环境：C:\\Python314\\python.exe（openpyxl 3.1.5）；.conda-env 无 openpyxl 时 build 会跳过 xlsx。
"""

from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

FILLABLE = {
    "评分0-4": ("0", "1", "2", "3", "4"),
    "人物_是否一致": ("是", "否", "无法判断"),
    "否定_是否一致": ("是", "否", "无法判断"),
    "数字_是否一致": ("是", "否", "无法判断"),
    "时间_是否一致": ("是", "否", "无法判断"),
    "动作_是否一致": ("是", "否", "无法判断"),
    "对象_是否一致": ("是", "否", "无法判断"),
    "严重语义错误": ("是", "否"),
    "流畅度1-5": ("1", "2", "3", "4", "5"),
    "可用于交流": ("是", "否", "无法判断"),
}
TEXT_FILLABLE = ("评审人", "评审日期", "备注", "复核人", "复核评分0-4", "复核意见")
WIDTHS = {
    "行号": 6, "review_id": 10, "参考译文_EVAL_ONLY_仅评审用": 42, "候选译文": 42,
    "评分0-4": 8, "人物_是否一致": 12, "否定_是否一致": 12, "数字_是否一致": 12,
    "时间_是否一致": 12, "动作_是否一致": 12, "对象_是否一致": 12, "严重语义错误": 12,
    "流畅度1-5": 10, "可用于交流": 12, "评审人": 12, "评审日期": 12, "备注": 30,
    "复核人": 10, "复核评分0-4": 11, "复核意见": 24,
}


def _instructions(ws) -> None:
    lines = [
        ("Round-2 人工语义评审 · 填写说明", None),
        ("评审对象", "A（Uni-Sign 视频）/ A2（stage1 官方 pose，当前 pose 默认）/ S2（stage2 pose 子集，候选）三条候选译文"),
        ("盲序", "模型身份已隐藏；同一样本的三个候选不相邻；不要查看 blind_map_EVAL_ONLY.csv"),
        ("填写方式", "在【评审表】黄色列逐行填写；灰色列（行号/review_id/参考译文/候选译文）只读"),
        ("保存", "另存为 round2_review_filled.xlsx，不要覆盖模板"),
        ("校验/统计", "C:\\Python314\\python.exe make_round2_package.py stats --review-file round2_review_filled.xlsx"),
        ("边界", "参考译文仅供判定语义，不得回灌模型/提示词；数据为 1086 子集，非新盲测"),
        ("评分0-4", "0 完全错误/相反；1 主要错误；2 部分正确但关键缺漏；3 核心意思正确、局部偏差；4 完整忠实自然"),
        ("六槽位", "与参考是否一致：人物/否定/数字/时间/动作/对象（是/否/无法判断）"),
        ("重点", "本轮核心是 A2 vs S2 的差异；若两者都能理解，请按事实错误程度区分"),
    ]
    ws.append(["项", "说明"])
    ws["A1"].font = Font(bold=True)
    ws["B1"].font = Font(bold=True)
    for key, value in lines:
        ws.append([key, value])
    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 110
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")


def _scale(ws) -> None:
    rows = [
        ("评分量表（0-4 语义忠实度）", ""),
        ("0", "完全错误 / 核心意思相反"),
        ("1", "主要错误：人物、动作、对象、否定、数字、时间等关键信息明显错误"),
        ("2", "部分正确：能看出部分意思，但关键信息缺漏或明显偏差"),
        ("3", "核心意思正确：局部表达/细节偏差，但不影响理解"),
        ("4", "语义完整忠实且表达自然"),
        ("六槽位", "人物 / 否定 / 数字 / 时间 / 动作 / 对象，各填 是 / 否 / 无法判断"),
        ("严重语义错误", "整句是否存在会导致误解的事实性错误（是/否）"),
        ("流畅度1-5", "1 不成句 … 5 完全自然"),
        ("可用于交流", "不看参考，这句话能否让人大致明白（是/否/无法判断）"),
    ]
    for row in rows:
        ws.append(list(row))
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 100


def write_workbook(csv_path: Path, xlsx_path: Path, columns: list[str]) -> dict:
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

    _instructions(wb.create_sheet("填写说明"))
    _scale(wb.create_sheet("评分量表"))
    progress = wb.create_sheet("进度统计")
    progress["A1"] = "已填写行数（自动）"
    progress["B1"] = f"=COUNT('评审表'!E2:E{len(rows) + 1})"
    progress["A2"] = "总行数"
    progress["B2"] = len(rows)
    progress["A3"] = "严重语义错误=是 的行数"
    progress["B3"] = f"=COUNTIF('评审表'!L2:L{len(rows) + 1},\"是\")"
    progress.column_dimensions["A"].width = 26
    progress.column_dimensions["B"].width = 12

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)
    return {"rows": len(rows), "columns": len(columns), "validations": validations}


if __name__ == "__main__":
    import sys
    csv_in, xlsx_out = Path(sys.argv[1]), Path(sys.argv[2])
    with open(csv_in, encoding="utf-8-sig") as fh:
        cols = next(csv.reader(fh))
    print(write_workbook(csv_in, xlsx_out, cols))

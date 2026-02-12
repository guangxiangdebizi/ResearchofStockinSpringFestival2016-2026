import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tushare as ts

DEFAULT_NAME_KEYWORDS = [
    "酒店",
    "旅游",
    "饮食",
    "餐饮",
    "景区",
    "免税",
    "白酒",
    "啤酒",
    "乳业",
    "乳品",
    "饮料",
    "食品",
    "调味",
    "零食",
    "卤味",
    "烘焙",
    "酿酒",
]

DEFAULT_INDUSTRY_KEYWORDS = [
    "酒店",
    "餐饮",
    "旅游",
    "白酒",
    "啤酒",
    "红黄酒",
    "软饮料",
    "乳制品",
    "食品",
    "农副食品",
]


@dataclass
class HolidayEvent:
    holiday_type: str
    pre_trade_date: pd.Timestamp
    post_trade_date: pd.Timestamp
    closure_start: pd.Timestamp
    closure_end: pd.Timestamp
    gap_days: int


def classify_holiday(closure_start: pd.Timestamp, gap_days: int) -> str:
    m = closure_start.month
    if m == 1 and closure_start.day <= 5 and gap_days <= 5:
        return "元旦"
    if m in (1, 2):
        return "春节"
    if m == 4:
        return "清明"
    if m == 5:
        return "五一"
    if m == 6:
        return "端午"
    if m in (9, 10):
        if gap_days >= 5 or m == 10:
            return "国庆"
        return "中秋"
    return "其他"


def detect_holiday_events(open_trade_dates: pd.DatetimeIndex) -> List[HolidayEvent]:
    events: List[HolidayEvent] = []
    dates = pd.Series(open_trade_dates).sort_values().reset_index(drop=True)

    for i in range(len(dates) - 1):
        prev_day = dates.iloc[i]
        next_day = dates.iloc[i + 1]
        gap_days = (next_day - prev_day).days - 1
        if gap_days < 3:
            continue

        closure_start = prev_day + pd.Timedelta(days=1)
        closure_end = next_day - pd.Timedelta(days=1)
        holiday_type = classify_holiday(closure_start, gap_days)

        events.append(
            HolidayEvent(
                holiday_type=holiday_type,
                pre_trade_date=prev_day,
                post_trade_date=next_day,
                closure_start=closure_start,
                closure_end=closure_end,
                gap_days=gap_days,
            )
        )

    return events


def fetch_trade_calendar(pro, start_date: str, end_date: str) -> pd.DatetimeIndex:
    cal = pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date, is_open="1")
    if cal is None or cal.empty:
        raise RuntimeError("无法获取交易日历")
    trade_days = pd.to_datetime(cal["cal_date"], format="%Y%m%d").sort_values().unique()
    return pd.DatetimeIndex(trade_days)


def fetch_daily_pct_chg(pro, ts_code: str, start_date: str, end_date: str, is_index: bool = False) -> pd.DataFrame:
    if is_index:
        df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    else:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        raise RuntimeError(f"无数据: {ts_code}")

    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date")

    if "pct_chg" not in df.columns:
        df["pct_chg"] = df["close"].pct_change() * 100

    return df[["trade_date", "pct_chg"]].dropna()


def build_thematic_stock_pool(
    pro,
    include_st: bool = True,
    include_bj: bool = False,
    extra_keywords: Optional[List[str]] = None,
) -> pd.DataFrame:
    fields = "ts_code,symbol,name,industry,market,list_date"
    df = pro.stock_basic(exchange="", list_status="L", fields=fields)
    if df is None or df.empty:
        raise RuntimeError("无法获取 stock_basic")

    df = df.copy()
    df["name"] = df["name"].fillna("")
    df["industry"] = df["industry"].fillna("")

    name_keywords = list(DEFAULT_NAME_KEYWORDS)
    if extra_keywords:
        name_keywords.extend([k for k in extra_keywords if k])
    industry_keywords = list(DEFAULT_INDUSTRY_KEYWORDS)

    def _hits(text: str, kws: List[str]) -> str:
        matched = [k for k in kws if k in text]
        return ",".join(matched)

    df["name_hits"] = df["name"].apply(lambda x: _hits(x, name_keywords))
    df["industry_hits"] = df["industry"].apply(lambda x: _hits(x, industry_keywords))

    mask = (df["name_hits"] != "") | (df["industry_hits"] != "")
    pool = df[mask].copy()

    if not include_bj:
        pool = pool[pool["ts_code"].str.endswith(".SH") | pool["ts_code"].str.endswith(".SZ")]

    if not include_st:
        pool = pool[~pool["name"].str.contains("ST", case=False, na=False)]

    pool["match_source"] = np.where(
        (pool["name_hits"] != "") & (pool["industry_hits"] != ""),
        "name+industry",
        np.where(pool["name_hits"] != "", "name", "industry"),
    )
    pool = pool.sort_values(["industry", "ts_code"]).reset_index(drop=True)
    return pool


def build_holiday_indicators(
    trade_days: pd.DatetimeIndex,
    events: List[HolidayEvent],
    pre_window: int = 7,
) -> Dict[str, pd.Series]:
    days = pd.Series(trade_days).sort_values().reset_index(drop=True)
    idx_map = {d: i for i, d in enumerate(days)}
    indicators: Dict[str, np.ndarray] = {}

    for e in events:
        if e.holiday_type in ("其他", "元旦"):
            continue
        indicators.setdefault(e.holiday_type, np.zeros(len(days), dtype=int))
        if e.pre_trade_date not in idx_map:
            continue
        end_idx = idx_map[e.pre_trade_date]
        start_idx = max(0, end_idx - pre_window + 1)
        indicators[e.holiday_type][start_idx : end_idx + 1] = 1

    return {k: pd.Series(v, index=days) for k, v in indicators.items()}


def window_cum_return(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return (np.prod(1 + series / 100.0) - 1) * 100


def stock_holiday_analysis(
    stock_returns: pd.Series,
    trade_days: pd.DatetimeIndex,
    events: List[HolidayEvent],
    pre_window: int = 7,
    post_window: int = 7,
) -> pd.DataFrame:
    days = pd.Series(trade_days).sort_values().reset_index(drop=True)
    idx_map = {d: i for i, d in enumerate(days)}

    indicators = build_holiday_indicators(trade_days=trade_days, events=events, pre_window=pre_window)

    sr = stock_returns.reindex(days)

    rows = []
    for holiday, ind in indicators.items():
        pair = pd.concat([sr.rename("ret"), ind.rename("flag")], axis=1).dropna()
        if len(pair) < 5 or pair["ret"].std(ddof=0) == 0 or pair["flag"].std(ddof=0) == 0:
            corr = np.nan
        else:
            corr = pair["ret"].corr(pair["flag"])

        per_event_pre = []
        per_event_post = []
        n_used = 0

        for e in events:
            if e.holiday_type != holiday:
                continue
            if e.pre_trade_date not in idx_map or e.post_trade_date not in idx_map:
                continue

            pre_end = idx_map[e.pre_trade_date]
            pre_start = max(0, pre_end - pre_window + 1)
            pre_dates = days.iloc[pre_start : pre_end + 1]

            post_start = idx_map[e.post_trade_date]
            post_end = min(len(days) - 1, post_start + post_window - 1)
            post_dates = days.iloc[post_start : post_end + 1]

            pre_ret = window_cum_return(sr.reindex(pre_dates).dropna())
            post_ret = window_cum_return(sr.reindex(post_dates).dropna())

            has_data = False
            if not np.isnan(pre_ret):
                per_event_pre.append(pre_ret)
                has_data = True
            if not np.isnan(post_ret):
                per_event_post.append(post_ret)
                has_data = True
            if has_data:
                n_used += 1

        rows.append(
            {
                "holiday": holiday,
                "corr_pre7_dummy": corr,
                "avg_pre7_cum_return_pct": np.nan if not per_event_pre else float(np.mean(per_event_pre)),
                "avg_post7_cum_return_pct": np.nan if not per_event_post else float(np.mean(per_event_post)),
                "events": n_used,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("corr_pre7_dummy", ascending=False)


def spring_festival_index_analysis(
    idx_returns: pd.Series,
    trade_days: pd.DatetimeIndex,
    events: List[HolidayEvent],
    pre_window: int = 7,
    post_window: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    days = pd.Series(trade_days).sort_values().reset_index(drop=True)
    idx_map = {d: i for i, d in enumerate(days)}
    sr = idx_returns.reindex(days)

    spring_events = [e for e in events if e.holiday_type == "春节"]

    year_rows = []
    path_rows = []

    for e in spring_events:
        if e.pre_trade_date not in idx_map or e.post_trade_date not in idx_map:
            continue

        pre_end = idx_map[e.pre_trade_date]
        pre_start = max(0, pre_end - pre_window + 1)
        pre_dates = days.iloc[pre_start : pre_end + 1]

        post_start = idx_map[e.post_trade_date]
        post_end = min(len(days) - 1, post_start + post_window - 1)
        post_dates = days.iloc[post_start : post_end + 1]

        pre_series = sr.reindex(pre_dates).dropna()
        post_series = sr.reindex(post_dates).dropna()

        year_rows.append(
            {
                "spring_year": int(e.closure_start.year),
                "holiday_start": e.closure_start.date().isoformat(),
                "holiday_end": e.closure_end.date().isoformat(),
                "pre7_cum_return_pct": window_cum_return(pre_series),
                "post7_cum_return_pct": window_cum_return(post_series),
            }
        )

        pre_list = list(pre_series.values)
        if len(pre_list) == pre_window:
            for i, r in enumerate(pre_list):
                rel_day = -pre_window + i
                path_rows.append({"spring_year": int(e.closure_start.year), "rel_day": rel_day, "ret_pct": float(r)})

        post_list = list(post_series.values)
        if len(post_list) == post_window:
            for i, r in enumerate(post_list):
                rel_day = i + 1
                path_rows.append({"spring_year": int(e.closure_start.year), "rel_day": rel_day, "ret_pct": float(r)})

    year_df = pd.DataFrame(year_rows).sort_values("spring_year")
    path_df = pd.DataFrame(path_rows)

    if not path_df.empty:
        avg_path = path_df.groupby("rel_day", as_index=False)["ret_pct"].mean().sort_values("rel_day")
        avg_path["cum_pct"] = (1 + avg_path["ret_pct"] / 100.0).cumprod().sub(1).mul(100)
    else:
        avg_path = pd.DataFrame(columns=["rel_day", "ret_pct", "cum_pct"])

    return year_df, avg_path


def write_df_sheet(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]
    rows, cols = df.shape

    ws.freeze_panes(1, 0)
    if cols > 0:
        ws.autofilter(0, 0, max(rows, 1), cols - 1)

    for c in range(cols):
        col_name = str(df.columns[c])
        if rows > 0:
            sample_len = int(df.iloc[:, c].astype(str).str.len().quantile(0.9))
        else:
            sample_len = 12
        width = max(len(col_name) + 2, sample_len + 2)
        width = min(max(width, 10), 34)
        ws.set_column(c, c, width)


def export_visual_report(
    outdir: str,
    pool_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    all_detail_df: pd.DataFrame,
    haoxiangni_detail: Optional[pd.DataFrame],
    spring_year_df: pd.DataFrame,
    spring_avg_path_df: pd.DataFrame,
) -> str:
    report_path = os.path.join(outdir, "holiday_visual_report.xlsx")

    summary_vis = summary_df.copy()
    summary_vis.insert(0, "rank", np.arange(1, len(summary_vis) + 1))
    summary_vis["stock_label"] = summary_vis["stock_name"] + "(" + summary_vis["ts_code"] + ")"

    detail_vis = all_detail_df.copy()
    if not detail_vis.empty:
        detail_vis = detail_vis.sort_values("corr_pre7_dummy", ascending=False)
        detail_vis.insert(0, "rank", np.arange(1, len(detail_vis) + 1))
        detail_vis["stock_label"] = detail_vis["stock_name"] + "(" + detail_vis["ts_code"] + ")"

    corr_matrix = pd.DataFrame()
    pre_matrix = pd.DataFrame()
    post_matrix = pd.DataFrame()
    if not all_detail_df.empty:
        corr_matrix = all_detail_df.pivot_table(
            index=["ts_code", "stock_name"],
            columns="holiday",
            values="corr_pre7_dummy",
            aggfunc="first",
        ).reset_index()
        pre_matrix = all_detail_df.pivot_table(
            index=["ts_code", "stock_name"],
            columns="holiday",
            values="avg_pre7_cum_return_pct",
            aggfunc="first",
        ).reset_index()
        post_matrix = all_detail_df.pivot_table(
            index=["ts_code", "stock_name"],
            columns="holiday",
            values="avg_post7_cum_return_pct",
            aggfunc="first",
        ).reset_index()

    with pd.ExcelWriter(report_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        write_df_sheet(writer, "股票池", pool_df)
        write_df_sheet(writer, "总览Top", summary_vis)
        write_df_sheet(writer, "个股节假日明细", detail_vis)
        write_df_sheet(writer, "春节逐年", spring_year_df)
        write_df_sheet(writer, "春节路径", spring_avg_path_df)
        write_df_sheet(writer, "相关性矩阵", corr_matrix)
        write_df_sheet(writer, "节前收益矩阵", pre_matrix)
        write_df_sheet(writer, "节后收益矩阵", post_matrix)
        if haoxiangni_detail is not None:
            write_df_sheet(writer, "好想你详情", haoxiangni_detail)

        ws_summary = writer.sheets["总览Top"]
        n_sum = len(summary_vis)
        if n_sum > 0:
            corr_col = summary_vis.columns.get_loc("best_corr_pre7_dummy")
            pre_col = summary_vis.columns.get_loc("best_avg_pre7_cum_return_pct")
            post_col = summary_vis.columns.get_loc("best_avg_post7_cum_return_pct")
            label_col = summary_vis.columns.get_loc("stock_label")
            evt_col = summary_vis.columns.get_loc("events")

            ws_summary.conditional_format(1, corr_col, n_sum, corr_col, {"type": "3_color_scale"})
            ws_summary.conditional_format(1, pre_col, n_sum, pre_col, {"type": "3_color_scale"})
            ws_summary.conditional_format(1, post_col, n_sum, post_col, {"type": "3_color_scale"})
            ws_summary.conditional_format(1, evt_col, n_sum, evt_col, {"type": "data_bar"})

            top_n = min(20, n_sum)
            chart = workbook.add_chart({"type": "column"})
            chart.add_series(
                {
                    "name": "节前7日相关系数",
                    "categories": ["总览Top", 1, label_col, top_n, label_col],
                    "values": ["总览Top", 1, corr_col, top_n, corr_col],
                }
            )
            chart.set_title({"name": "相关性Top20"})
            chart.set_legend({"none": True})
            chart.set_y_axis({"name": "corr"})
            ws_summary.insert_chart(1, len(summary_vis.columns) + 1, chart, {"x_scale": 1.5, "y_scale": 1.2})

        ws_detail = writer.sheets["个股节假日明细"]
        n_det = len(detail_vis)
        if n_det > 0:
            det_corr_col = detail_vis.columns.get_loc("corr_pre7_dummy")
            det_pre_col = detail_vis.columns.get_loc("avg_pre7_cum_return_pct")
            det_post_col = detail_vis.columns.get_loc("avg_post7_cum_return_pct")
            det_evt_col = detail_vis.columns.get_loc("events")

            ws_detail.conditional_format(1, det_corr_col, n_det, det_corr_col, {"type": "3_color_scale"})
            ws_detail.conditional_format(1, det_pre_col, n_det, det_pre_col, {"type": "3_color_scale"})
            ws_detail.conditional_format(1, det_post_col, n_det, det_post_col, {"type": "3_color_scale"})
            ws_detail.conditional_format(1, det_evt_col, n_det, det_evt_col, {"type": "data_bar"})

        for sheet_name in ["相关性矩阵", "节前收益矩阵", "节后收益矩阵"]:
            ws = writer.sheets[sheet_name]
            df_now = {"相关性矩阵": corr_matrix, "节前收益矩阵": pre_matrix, "节后收益矩阵": post_matrix}[sheet_name]
            if not df_now.empty and df_now.shape[1] > 2:
                ws.conditional_format(1, 2, len(df_now), df_now.shape[1] - 1, {"type": "3_color_scale"})

        ws_path = writer.sheets["春节路径"]
        n_path = len(spring_avg_path_df)
        if n_path > 0:
            day_col = spring_avg_path_df.columns.get_loc("rel_day")
            ret_col = spring_avg_path_df.columns.get_loc("ret_pct")
            cum_col = spring_avg_path_df.columns.get_loc("cum_pct")
            line = workbook.add_chart({"type": "line"})
            line.add_series(
                {
                    "name": "平均日收益(%)",
                    "categories": ["春节路径", 1, day_col, n_path, day_col],
                    "values": ["春节路径", 1, ret_col, n_path, ret_col],
                }
            )
            line.add_series(
                {
                    "name": "累计收益(%)",
                    "categories": ["春节路径", 1, day_col, n_path, day_col],
                    "values": ["春节路径", 1, cum_col, n_path, cum_col],
                    "y2_axis": True,
                }
            )
            line.set_title({"name": "春节前后7日平均路径"})
            line.set_x_axis({"name": "相对交易日"})
            line.set_y_axis({"name": "日收益(%)"})
            line.set_y2_axis({"name": "累计收益(%)"})
            ws_path.insert_chart(1, len(spring_avg_path_df.columns) + 1, line, {"x_scale": 1.6, "y_scale": 1.2})

    return report_path


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_无数据_"
    show = df.head(max_rows).copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    cols = [str(c) for c in show.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in show.iterrows():
        vals = [str(v) if not pd.isna(v) else "" for v in row.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def export_markdown_report(
    outdir: str,
    start_date: str,
    end_date: str,
    pool_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    all_detail_df: pd.DataFrame,
    haoxiangni_detail: Optional[pd.DataFrame],
    spring_year_df: pd.DataFrame,
    spring_avg_path_df: pd.DataFrame,
) -> str:
    report_md_path = os.path.join(outdir, "REPORT.md")

    industry_top = pool_df["industry"].fillna("").replace("", "未知").value_counts().head(15).reset_index()
    industry_top.columns = ["industry", "count"]

    spring_stats = {}
    if not spring_year_df.empty:
        spring_stats = {
            "years": len(spring_year_df),
            "pre_mean": float(spring_year_df["pre7_cum_return_pct"].mean()),
            "post_mean": float(spring_year_df["post7_cum_return_pct"].mean()),
            "pre_win_rate": float((spring_year_df["pre7_cum_return_pct"] > 0).mean()),
            "post_win_rate": float((spring_year_df["post7_cum_return_pct"] > 0).mean()),
        }

    by_holiday = pd.DataFrame()
    if not all_detail_df.empty:
        by_holiday = (
            all_detail_df.groupby("holiday", as_index=False)
            .agg(
                mean_corr=("corr_pre7_dummy", "mean"),
                median_corr=("corr_pre7_dummy", "median"),
                mean_pre7=("avg_pre7_cum_return_pct", "mean"),
                mean_post7=("avg_post7_cum_return_pct", "mean"),
                stocks=("ts_code", "nunique"),
            )
            .sort_values("mean_corr", ascending=False)
        )

    top_cols = [
        "ts_code",
        "stock_name",
        "best_holiday",
        "best_corr_pre7_dummy",
        "best_avg_pre7_cum_return_pct",
        "best_avg_post7_cum_return_pct",
        "events",
    ]
    hx_cols = [
        "holiday",
        "corr_pre7_dummy",
        "avg_pre7_cum_return_pct",
        "avg_post7_cum_return_pct",
        "events",
    ]

    lines = []
    lines.append("# 节假日题材股票研究报告")
    lines.append("")
    lines.append("## 1. 研究范围")
    lines.append(f"- 数据源: TuShare 日线行情 (`pro.daily`, `pro.index_daily`, `pro.trade_cal`)")
    lines.append(f"- 区间: `{start_date}` - `{end_date}`")
    lines.append("- 标的范围: 自动筛选酒店/旅游/餐饮/食品/饮料/酒类等相关个股（含`好想你`）")
    lines.append(f"- 股票池数量: `{len(pool_df)}`")
    lines.append(f"- 成功分析数量: `{len(summary_df)}`")
    lines.append("")
    lines.append("## 2. 股票池行业分布（Top15）")
    lines.append(markdown_table(industry_top, max_rows=15))
    lines.append("")
    lines.append("## 3. 节前7日相关性最高个股（Top20）")
    lines.append(markdown_table(summary_df[top_cols], max_rows=20))
    lines.append("")
    lines.append("## 4. 各节假日整体统计（全样本）")
    lines.append(markdown_table(by_holiday, max_rows=20))
    lines.append("")
    lines.append("## 5. 好想你（002582.SZ）明细")
    if haoxiangni_detail is None or haoxiangni_detail.empty:
        lines.append("_未获取到好想你明细数据_")
    else:
        lines.append(markdown_table(haoxiangni_detail[hx_cols], max_rows=20))
    lines.append("")
    lines.append("## 6. 春节前后大盘（上证指数）")
    if spring_stats:
        lines.append(f"- 样本年份数: `{spring_stats['years']}`")
        lines.append(f"- 春节前7日平均累计收益: `{spring_stats['pre_mean']:.4f}%`")
        lines.append(f"- 春节后7日平均累计收益: `{spring_stats['post_mean']:.4f}%`")
        lines.append(f"- 春节前7日胜率: `{spring_stats['pre_win_rate']:.2%}`")
        lines.append(f"- 春节后7日胜率: `{spring_stats['post_win_rate']:.2%}`")
    lines.append("")
    lines.append("### 春节逐年表现")
    lines.append(markdown_table(spring_year_df, max_rows=20))
    lines.append("")
    lines.append("### 春节平均路径（日收益与累计）")
    lines.append(markdown_table(spring_avg_path_df, max_rows=20))
    lines.append("")
    lines.append("## 7. 文件清单")
    lines.append("- `thematic_stock_pool.csv`: 自动筛选的股票池")
    lines.append("- `holiday_correlation_summary.csv`: 每只股票最强节假日汇总")
    lines.append("- `all_stocks_holiday_detail.csv`: 每只股票 x 每个节假日完整明细")
    lines.append("- `haoxiangni_holiday_detail.csv`: 好想你明细")
    lines.append("- `spring_festival_index_7d_by_year.csv`: 春节逐年统计")
    lines.append("- `spring_festival_index_7d_avg_path.csv`: 春节平均路径")
    lines.append("- `holiday_visual_report.xlsx`: 可视化表格报告")
    lines.append("- `REPORT.md`: 本报告")
    lines.append("")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Holiday related stock analysis based on TuShare daily data")
    parser.add_argument("--token", default=os.getenv("TUSHARE_TOKEN"), help="TuShare token")
    parser.add_argument("--start", default="20160101")
    parser.add_argument("--end", default="20260212")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--exclude-st", action="store_true", help="排除 ST/*ST")
    parser.add_argument("--include-bj", action="store_true", help="包含北交所 .BJ")
    parser.add_argument("--max-stocks", type=int, default=0, help="最多分析多少只，0=不限制")
    parser.add_argument("--extra-keywords", default="", help="额外名称关键词，逗号分隔")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("请提供 TuShare token（--token 或 TUSHARE_TOKEN）")

    os.makedirs(args.outdir, exist_ok=True)

    ts.set_token(args.token)
    pro = ts.pro_api()
    extra_keywords = [x.strip() for x in args.extra_keywords.split(",") if x.strip()]
    pool_df = build_thematic_stock_pool(
        pro,
        include_st=not args.exclude_st,
        include_bj=args.include_bj,
        extra_keywords=extra_keywords,
    )
    if args.max_stocks and len(pool_df) > args.max_stocks:
        pool_df = pool_df.head(args.max_stocks).copy()

    if "002582.SZ" not in set(pool_df["ts_code"]):
        extra_row = pd.DataFrame(
            [
                {
                    "ts_code": "002582.SZ",
                    "symbol": "002582",
                    "name": "好想你",
                    "industry": "食品",
                    "market": "",
                    "list_date": "",
                    "name_hits": "好想你",
                    "industry_hits": "食品",
                    "match_source": "manual",
                }
            ]
        )
        pool_df = pd.concat([pool_df, extra_row], ignore_index=True)

    pool_path = os.path.join(args.outdir, "thematic_stock_pool.csv")
    pool_df.to_csv(pool_path, index=False, encoding="utf-8-sig")

    trade_days = fetch_trade_calendar(pro, args.start, args.end)
    events = detect_holiday_events(trade_days)

    all_rows = []
    all_detail_rows = []
    haoxiangni_detail = None

    total = len(pool_df)
    for i, row in pool_df.iterrows():
        code = row["ts_code"]
        name = row["name"]
        try:
            df = fetch_daily_pct_chg(pro, code, args.start, args.end, is_index=False)
        except Exception:
            continue
        ret = df.set_index("trade_date")["pct_chg"]

        detail = stock_holiday_analysis(ret, trade_days, events, pre_window=7, post_window=7)
        detail = detail[detail["events"] > 0].copy()
        if detail.empty:
            continue

        detail.insert(0, "stock_name", name)
        detail.insert(0, "ts_code", code)
        all_detail_rows.append(detail.copy())

        if code == "002582.SZ":
            haoxiangni_detail = detail.copy()

        best = detail.iloc[0].to_dict()
        all_rows.append(
            {
                "ts_code": code,
                "stock_name": name,
                "best_holiday": best["holiday"],
                "best_corr_pre7_dummy": best["corr_pre7_dummy"],
                "best_avg_pre7_cum_return_pct": best["avg_pre7_cum_return_pct"],
                "best_avg_post7_cum_return_pct": best["avg_post7_cum_return_pct"],
                "events": int(best["events"]),
            }
        )

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"分析进度: {i + 1}/{total}")
        time.sleep(0.12)

    summary_df = pd.DataFrame(all_rows)
    if summary_df.empty:
        raise RuntimeError("股票池分析结果为空，请调整关键词或时间区间后重试")
    summary_df = summary_df.sort_values("best_corr_pre7_dummy", ascending=False)
    summary_path = os.path.join(args.outdir, "holiday_correlation_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    all_detail_df = pd.DataFrame()
    if all_detail_rows:
        all_detail_df = pd.concat(all_detail_rows, ignore_index=True)
        all_detail_df.to_csv(os.path.join(args.outdir, "all_stocks_holiday_detail.csv"), index=False, encoding="utf-8-sig")

    if haoxiangni_detail is not None:
        hx_path = os.path.join(args.outdir, "haoxiangni_holiday_detail.csv")
        haoxiangni_detail.to_csv(hx_path, index=False, encoding="utf-8-sig")

    idx_df = fetch_daily_pct_chg(pro, "000001.SH", args.start, args.end, is_index=True)
    idx_ret = idx_df.set_index("trade_date")["pct_chg"]

    spring_year_df, spring_avg_path_df = spring_festival_index_analysis(idx_ret, trade_days, events, pre_window=7, post_window=7)

    spring_year_path = os.path.join(args.outdir, "spring_festival_index_7d_by_year.csv")
    spring_avg_path = os.path.join(args.outdir, "spring_festival_index_7d_avg_path.csv")
    spring_year_df.to_csv(spring_year_path, index=False, encoding="utf-8-sig")
    spring_avg_path_df.to_csv(spring_avg_path, index=False, encoding="utf-8-sig")

    visual_report_path = export_visual_report(
        outdir=args.outdir,
        pool_df=pool_df,
        summary_df=summary_df,
        all_detail_df=all_detail_df,
        haoxiangni_detail=haoxiangni_detail,
        spring_year_df=spring_year_df,
        spring_avg_path_df=spring_avg_path_df,
    )
    report_md_path = export_markdown_report(
        outdir=args.outdir,
        start_date=args.start,
        end_date=args.end,
        pool_df=pool_df,
        summary_df=summary_df,
        all_detail_df=all_detail_df,
        haoxiangni_detail=haoxiangni_detail,
        spring_year_df=spring_year_df,
        spring_avg_path_df=spring_avg_path_df,
    )

    print("=== 输出文件 ===")
    print(pool_path)
    print(summary_path)
    print(os.path.join(args.outdir, "haoxiangni_holiday_detail.csv"))
    print(spring_year_path)
    print(spring_avg_path)
    print(visual_report_path)
    print(report_md_path)

    print(f"\n股票池数量: {len(pool_df)}，成功分析数量: {len(summary_df)}")

    print("\n=== 节假日相关性 Top20 ===")
    print(summary_df.head(20).to_string(index=False))

    if haoxiangni_detail is not None:
        print("\n=== 好想你 详情 ===")
        print(haoxiangni_detail.to_string(index=False))

    if not spring_year_df.empty:
        print("\n=== 春节大盘 7日（逐年） ===")
        print(spring_year_df.to_string(index=False))

        print("\n=== 春节大盘 平均路径（日收益与累计） ===")
        print(spring_avg_path_df.to_string(index=False))


if __name__ == "__main__":
    main()

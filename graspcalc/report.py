"""PDF report of one or more score_parts runs: method, per-category tables, box plots, per-object visualisations.

python -m graspcalc.report --runs out_parts_teapot_handle out_parts_mug_handle --out report.pdf
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

from .summarize import INTENTS, boxes, load

CAT_CN = {"teapot": "茶壶", "mug": "杯子"}
PART_CN = {"handle": "把手", "body": "主体", "spout": "壶嘴", "lid": "盖"}
for _f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
    if any(x.name == _f for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
W, H = 11.69, 8.27


def page(title):
    fig = plt.figure(figsize=(W, H))
    fig.text(0.05, 0.95, title, fontsize=16, weight="bold", va="top")
    return fig


def flush(pdf, fig):
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def table(fig, rect, header, rows, fs=9):
    ax = fig.add_axes(rect)
    ax.axis("off")
    if not rows:
        return
    t = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(fs)
    t.scale(1, 1.35)
    for (r, _), c in t.get_celld().items():
        if r == 0:
            c.set_facecolor("#dde6f3")
            c.set_text_props(weight="bold")


def cls_cn(c):
    return "无接触" if c == "no contact" else "+".join(PART_CN.get(p, p) for p in c.split("+"))


def retrieval_rows(retr, aff):
    if not retr:
        return []
    sh = np.array([sum(r["ref_part_share"].get(p, 0) for p in aff) for r in retr]) if aff else np.ones(len(retr))
    groups = [("全部", sh >= -1)]
    if aff:
        name = "+".join(PART_CN.get(p, p) for p in aff)
        groups = [(f"参考碰到{name}（≥10%）", sh >= 0.1), (f"参考基本不碰{name}（<10%）", sh < 0.1)] + groups
    out = []
    for name, sel in groups:
        r = [x for x, s in zip(retr, sel) if s]
        if r:
            w = np.array([x["n_objects"] for x in r], float)
            avg = lambda k: (np.array([x[k] for x in r]) * w).sum() / w.sum()
            out.append([name, len(r), int(w.sum()), f"{avg('mean_n'):.0f}", f"{avg('top1') * 100:.1f}%", f"{avg('mean_rank'):.2f}"])
    return out


def method_page(pdf, runs):
    m0 = runs[0][0]
    fig = page("灵巧手抓取评分：单一参考、按功能部件比较接触")
    refs = []
    for m, rows, _ in runs:
        ref = next(r for r in rows if r["is_ref"])
        share = " / ".join(f"{PART_CN.get(k, k)} {v * 100:.0f}%"
                           for k, v in sorted(ref["part_share"].items(), key=lambda kv: -kv[1]) if v > 0.01)
        refs.append(f"{CAT_CN.get(m['cat'], m['cat'])} {m['ref']}（{ref['intent']}，{share}），{m['n_objects']} 个物体"
                    + (f"，跳过 {', '.join(m['skipped'])}（OakBase 不对齐或缺失）" if m["skipped"] else ""))
    aff = m0["afford_parts"]
    txt = [
        "目标：给任意物体实例上的抓取打分，越接近参考抓取（OakInk 的 MANO 功能性抓取）分数越高；手通过统一接口接入，可换成任意 URDF 灵巧手。",
        "",
        "手的统一表示（HandState）：带 16 个语义部位标签的表面点（手掌 + 5 指 × 3 节）、手腕、每根手指的关键点链。",
        "",
        "F1 手型：掌心坐标系下的指尖位置、弯曲角、张开角、伸直程度、拇指到各指尖的距离；F1 = exp(−RMS(归一化误差) / T)，"
        f"T 放宽 {m0['f1_loosen']} 倍。",
        "",
        "F4p 部件接触：物体采样点通过最近邻继承 OakBase 部件标签；手部位 k 对物体点 o 的软接触 exp(−(d_align/2cm)²)，",
        "        按部件求和并归一化，得到 16 × P 的接触占比矩阵（与物体实例无关，可跨物体比较）。",
        (f"        只看 affordance 部件 {aff}：取这些部件上的 16 维接触分布，归一化后与参考算软 IoU，其他部件不参与。" if aff else
         "        F4p = sqrt(部件占比的软 IoU × 16×P 矩阵的软 IoU)。"),
        "",
        f"总分：S = F1 × sigmoid((F4p − {m0['gate_tau']}) / {m0['gate_width']})。"
        + ("无部件占比硬条件。" if m0["min_part_ratio"] <= 0 else
           f"硬条件：参考占比 ≥{m0['ref_part_min']} 的部件，候选占比 < {m0['min_part_ratio']} × 参考时记 0。"),
        *(["        只看 affordance 时再乘连续折扣 min(候选在 affordance 上的接触占比 / 参考的占比, 1)，"
            "防止 affordance 上的微弱接触被归一化放大。"] if aff and m0.get("share_discount", 0) else []),
        "        不使用物体对称和镜像，左右手区分。",
        "",
        "参考（每个类别一个参考，给该类别所有实例上的所有抓取打分，意图不参与打分）：",
    ] + [f"        {r}" for r in refs] + [
        "",
        "跨物体检索检验：用真实物体上的一个抓取当参考，在每个虚拟实例上，它迁移过去的那份抓取应排第 1。",
    ]
    fig.text(0.05, 0.88, "\n".join(txt), fontsize=10, va="top", linespacing=1.55)
    flush(pdf, fig)


def category_pages(pdf, m, rows, retr, run_dir):
    cn = CAT_CN.get(m["cat"], m["cat"])
    R = load(os.path.join(run_dir, "part_scores.json"))
    fig = page(f"{cn}（{m['cat']}）：统计结果    参考 {m['ref']}")
    fig.text(0.05, 0.89, f"{m['n_objects']} 个物体，{len(R)} 个候选抓取（不含参考本身）。百分位：物体内名次换算，1 = 第 1 名。",
             fontsize=11, va="top")
    classes = sorted(set(r["cls"] for r in R), key=lambda c: -np.mean([r["total"] for r in R if r["cls"] == c]))
    trows = []
    for c in classes:
        x = [r for r in R if r["cls"] == c]
        t = np.array([r["total"] for r in x])
        mm = lambda k: np.mean([r[k] for r in x])
        trows.append([cls_cn(c), len(x), f"{t.mean():.3f}", f"{np.median(t):.3f}", f"[{t.min():.2f}, {t.max():.2f}]",
                      f"{mm('s1_shape'):.2f}", f"{mm('s4p'):.2f}", f"{mm('pct'):.2f}",
                      sum(r["rank"] == 1 for r in x), sum(r["rank"] == r["n"] for r in x)])
    fig.text(0.05, 0.84, "按接触部件分组（占总接触 >15% 的部件，按占比从大到小）", fontsize=12, weight="bold")
    table(fig, [0.05, 0.50, 0.9, 0.32],
          ["接触部件", "数量", "总分均值", "中位数", "[最小, 最大]", "F1", "F4p", "物体内百分位", "第 1 名", "最后一名"], trows[:10])
    fig.text(0.05, 0.45, "跨物体检索（参考 = 真实物体上的抓取，看它迁移到虚拟实例后的名次）", fontsize=12, weight="bold")
    aff = [p for p in m["afford_parts"].split(",") if p]
    table(fig, [0.05, 0.22, 0.9, 0.21], ["参考类型", "参考数", "检验对数", "每个物体候选数", "top-1", "平均名次"],
          retrieval_rows(retr, aff))
    best = sorted(R, key=lambda r: -r["total"])[:4]
    fig.text(0.05, 0.17, "得分最高：" + "；".join(f"{r['obj']}/{r['gid'][:6]} {r['total']:.2f}" for r in best), fontsize=9.5)
    fig.text(0.05, 0.13, f"总分 < 0.005 的抓取：{sum(r['total'] < 0.005 for r in R)} / {len(R)}", fontsize=9.5)
    flush(pdf, fig)

    title = f"{m['cat']} vs ONE reference {m['ref']}"
    for field, groups, f, t in (("cls", classes, "part_summary.png", "按接触部件分组的分数分布（点颜色 = 意图，仅作参考）"),
                                ("intent", INTENTS, "intent_summary.png", "按意图分组的分数分布（仅作参考）")):
        p = os.path.join(run_dir, f)
        boxes(R, field, groups, p, title)
        fig = page(f"{cn}：{t}")
        ax = fig.add_axes([0.02, 0.05, 0.96, 0.82])
        ax.imshow(imread(p))
        ax.axis("off")
        flush(pdf, fig)

    oids = list(dict.fromkeys(r["obj_id"] for r in rows))
    for i in range(0, len(oids), 3):
        fig = page(f"{cn}：逐物体结果（{i + 1}–{min(i + 3, len(oids))} / {len(oids)}）  第 1 格为参考，其后为最好 / 中位 / 最差的抓取")
        for j, oid in enumerate(oids[i:i + 3]):
            ax = fig.add_axes([0.01, 0.62 - j * 0.30, 0.98, 0.29])
            ax.imshow(imread(os.path.join(run_dir, f"viz_{oid}.png")))
            ax.axis("off")
        flush(pdf, fig)


def main(a):
    runs = []
    for d in a.runs:
        m = json.load(open(os.path.join(d, "meta.json")))
        rows = json.load(open(os.path.join(d, "part_scores.json")))
        rp = os.path.join(d, "retrieval.json")
        runs.append((m, rows, json.load(open(rp)) if os.path.exists(rp) else []))
    with PdfPages(a.out) as pdf:
        method_page(pdf, runs)
        for (m, rows, retr), d in zip(runs, a.runs):
            category_pages(pdf, m, rows, retr, d)
    print(f"saved {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="score_parts output directories")
    ap.add_argument("--out", default="grasp_score_report.pdf")
    main(ap.parse_args())

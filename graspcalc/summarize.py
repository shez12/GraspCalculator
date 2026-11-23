"""Summaries of score_parts.py output by intent and by contacted object parts (+ optional comparison to an older run)."""
import argparse
import collections
import json
import os

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INTENTS = ["use", "liftup", "handover", "hold"]
COLS = {"use": "tab:blue", "liftup": "tab:orange", "handover": "tab:green", "hold": "tab:red"}
KEYS = [("total", "total"), ("s1_shape", "F1 hand shape"), ("s4p", "F4p part contact")]


def load(path):
    R = [r for r in json.load(open(path)) if not r["is_ref"]]
    by = collections.defaultdict(list)
    for r in R:
        by[r["obj_id"]].append(r)
    for rs in by.values():
        rs.sort(key=lambda r: -r["total"])
        for k, r in enumerate(rs):
            r["rank"], r["n"], r["pct"] = k + 1, len(rs), 1 - k / max(len(rs) - 1, 1)
    for r in R:
        s = r["part_share"]
        r["cls"] = "no contact" if sum(s.values()) < 1e-6 else \
            "+".join(k for k, v in sorted(s.items(), key=lambda kv: -kv[1]) if v > 0.15)
    return R


def table(R, field, groups, title):
    print(f"\n== by {title} ==")
    print(f"{title:<16}{'n':>4} | {'total mean/med [min,max]':>26} | {'F1':>5} {'F4p':>5} {'gate':>5} {'ok%':>5} | "
          f"{'pct':>5} {'top1':>4} {'last':>4} | members")
    other = "cls" if field == "intent" else "intent"
    for c in groups:
        x = [r for r in R if r[field] == c]
        if not x:
            continue
        t = np.array([r["total"] for r in x])
        m = lambda k: np.mean([r.get(k, np.nan) for r in x])
        mem = collections.Counter(r[other] for r in x)
        print(f"{c:<16}{len(x):>4} | {t.mean():.3f}/{np.median(t):.3f} [{t.min():.2f},{t.max():.2f}]   | "
              f"{m('s1_shape'):>5.2f} {m('s4p'):>5.2f} {m('gate'):>5.2f} {100 * m('part_ok'):>5.0f} | {m('pct'):>5.2f} "
              f"{sum(r['rank'] == 1 for r in x):>4} {sum(r['rank'] == r['n'] for r in x):>4} | "
              + ", ".join(f"{k} {v}" for k, v in mem.most_common()))
    for c in groups:
        x = sorted([r for r in R if r[field] == c], key=lambda r: -r["total"])
        if x:
            f = lambda r: f"{r['obj']}/{r['gid'][:6]}({r[other]}) {r['total']:.2f}"
            print(f"  {c}: top " + "; ".join(f(r) for r in x[:3]) + "  | bottom " + "; ".join(f(r) for r in x[-2:]))


def boxes(R, field, groups, path, title):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for a, (k, lab) in zip(ax, KEYS):
        g = [c for c in groups if any(r[field] == c for r in R)]
        a.boxplot([[r[k] for r in R if r[field] == c] for c in g], showmeans=True,
                  tick_labels=[f"{c}\n(n={sum(r[field] == c for r in R)})" for c in g])
        for j, c in enumerate(g):
            x = [r for r in R if r[field] == c]
            a.scatter(np.random.normal(j + 1, 0.07, len(x)), [r[k] for r in x], s=8, alpha=0.6,
                      c=[COLS[r["intent"]] for r in x])
        a.set_title(lab)
        a.set_ylim(0, 1)
        a.tick_params(axis="x", labelsize=8)
    for i, cl in COLS.items():
        ax[0].scatter([], [], c=cl, label=i)
    ax[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(a):
    R = load(a.scores)
    classes = sorted(set(r["cls"] for r in R), key=lambda c: -np.mean([r["total"] for r in R if r["cls"] == c]))
    table(R, "intent", INTENTS, "intent")
    table(R, "cls", classes, "contact parts")
    d = os.path.dirname(a.scores)
    boxes(R, "intent", INTENTS, os.path.join(d, "intent_summary.png"), f"{a.title}: grouped by intent")
    boxes(R, "cls", classes, os.path.join(d, "part_summary.png"),
          f"{a.title}: grouped by contacted parts (>15% of contact)")
    if a.compare:
        old = {(r["obj_id"], r["gid"]): r for r in load(a.compare)}
        by = collections.defaultdict(list)
        for r in R:
            by[r["obj_id"]].append(r)
        rho, top_changed = [], []
        for oid, rs in by.items():
            if len(rs) > 2:
                rho.append(spearmanr([r["total"] for r in rs], [old[(oid, r["gid"])]["total"] for r in rs]).correlation)
            n1 = next(r for r in rs if r["rank"] == 1)
            o1 = next(old[(oid, r["gid"])] for r in rs if old[(oid, r["gid"])]["rank"] == 1)
            if n1["gid"] != o1["gid"]:
                top_changed.append(f"{n1['obj']}: {o1['gid'][:6]}({o1['intent']},{o1['cls']}) -> "
                                   f"{n1['gid'][:6]}({n1['intent']},{n1['cls']})")
        print(f"\n== vs {a.compare} ==\nper-object Spearman mean {np.nanmean(rho):.3f}, min {np.nanmin(rho):.3f}; "
              f"top-1 changed on {len(top_changed)} objects")
        for t in top_changed:
            print("  " + t)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--compare", default="")
    ap.add_argument("--title", default="scores")
    main(ap.parse_args())

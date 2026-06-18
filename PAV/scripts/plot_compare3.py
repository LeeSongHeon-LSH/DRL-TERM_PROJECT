import json, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("outputs")
RUNS = {
    "pav (Q3 dist)":          ("PAV-distribution-test",          "tab:blue"),
    "pav-fewshot (Q3 dist)":  ("PAV-distribution-fewshot-test",  "tab:green"),
    "pav-scalar-c2 (Q1)":     ("PAV-scalar-c2-test",             "tab:red"),
}
OUT = BASE / "comparison"
OUT.mkdir(parents=True, exist_ok=True)
CLIP = 500  # fair comparison window (fewshot only reaches 500)

def load(run):
    recs=[]
    for line in (BASE/run/"metrics.jsonl").read_text().splitlines():
        line=line.strip()
        if not line: continue
        try:
            r=json.loads(line)
            if r.get("step",10**9) <= CLIP: recs.append(r)
        except: pass
    return recs

def series(recs, key):
    xs=[r["step"] for r in recs if key in r and r[key] is not None]
    ys=[r[key]   for r in recs if key in r and r[key] is not None]
    return xs, ys

def roll(xs, ys, w=15):
    if not ys: return xs, ys
    out=[]
    for i in range(len(ys)):
        lo=max(0,i-w//2); hi=min(len(ys),i+w//2+1)
        out.append(sum(ys[lo:hi])/(hi-lo))
    return xs, out

def wmean(recs, lo, hi, key):
    vs=[r[key] for r in recs if lo<=r.get("step",-1)<=hi and key in r and r[key] is not None]
    return statistics.mean(vs) if vs else float("nan")

DATA={name: load(run) for name,(run,_) in RUNS.items()}

# ---------- numeric tables ----------
print("RUN RANGES")
for name,(run,_) in RUNS.items():
    steps=[r["step"] for r in DATA[name]]
    print(f"  {name:24s} {run:32s} steps {min(steps)}..{max(steps)} (n={len(steps)})")

CPS=[1,100,200,300,400,500]
METRICS=["reward","reward_std","kl","grad_norm","completion_length"]
for key in METRICS:
    print(f"\n## {key}  (windowed mean [cp-19,cp])")
    print("| step | " + " | ".join(RUNS) + " |")
    print("|---|" + "---|"*len(RUNS))
    for cp in CPS:
        lo=cp-19 if cp>1 else 1
        cells=[]
        for name in RUNS:
            v=wmean(DATA[name], lo, cp, key)
            cells.append(f"{v:.4f}" if key in ("reward","reward_std","grad_norm") else (f"{v:.5f}" if key=="kl" else f"{v:.1f}"))
        print(f"| {cp} | " + " | ".join(cells) + " |")

print("\n## FINAL (last logged step <=500, raw)")
print("| metric | " + " | ".join(RUNS) + " |")
print("|---|" + "---|"*len(RUNS))
last={name: DATA[name][-1] for name in RUNS}
for key in ["step"]+METRICS+["learning_rate"]:
    cells=[f'{last[name].get(key,float("nan")):.5g}' for name in RUNS]
    print(f"| {key} | " + " | ".join(cells) + " |")

print("\n## START(1-20)->END(481-500) delta (windowed)")
for name in RUNS:
    print(f"### {name}")
    for key in METRICS:
        s=wmean(DATA[name],1,20,key); e=wmean(DATA[name],481,500,key)
        print(f"  {key:18s} {s:9.4f} -> {e:9.4f}  (delta {e-s:+.4f})")

# ---------- comparison figure ----------
PANELS=[("reward","reward (PAV)"),("reward_std","reward_std"),("kl","KL(pi||pi_ref)"),
        ("grad_norm","grad_norm"),("completion_length","completion length (tok)"),("learning_rate","learning rate")]
fig,axes=plt.subplots(2,3,figsize=(16,9))
for ax,(key,title) in zip(axes.flat,PANELS):
    for name,(run,color) in RUNS.items():
        xs,ys=series(DATA[name],key)
        if not xs: continue
        ax.plot(xs,ys,color=color,alpha=0.12,linewidth=0.8)
        rx,ry=roll(xs,ys,15)
        ax.plot(rx,ry,color=color,linewidth=1.8,label=name)
    ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=0.3)
    if key=="reward": ax.legend(fontsize=8,loc="best")
fig.suptitle("PAV-RL training comparison — 3 runs (raw=faint, rolling-mean w15=bold), step<=500",fontsize=13)
fig.tight_layout()
fig.savefig(OUT/"overview_compare.png",dpi=120); plt.close(fig)
print(f"\nsaved {OUT/'overview_compare.png'}")

# individual key-metric compare plots (bigger, for embedding)
for key,title,fname in [("reward","Reward (PAV) — higher=better, NOT cross-reducer comparable","reward_compare.png"),
                        ("kl","KL(pi||pi_ref) — policy movement from reference","kl_compare.png"),
                        ("completion_length","Completion length (tokens)","completion_compare.png"),
                        ("grad_norm","Gradient L2 norm","grad_norm_compare.png")]:
    fig,ax=plt.subplots(figsize=(9,4.5))
    for name,(run,color) in RUNS.items():
        xs,ys=series(DATA[name],key)
        if not xs: continue
        ax.plot(xs,ys,color=color,alpha=0.12,linewidth=0.8)
        rx,ry=roll(xs,ys,15)
        ax.plot(rx,ry,color=color,linewidth=2.0,label=name)
    ax.set_title(title); ax.set_xlabel("step"); ax.set_ylabel(key); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/fname,dpi=120); plt.close(fig)
    print(f"saved {OUT/fname}")

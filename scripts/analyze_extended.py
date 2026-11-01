from pathlib import Path
import json,csv,collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/"results/deep_analysis";O.mkdir(exist_ok=True)
D=R/"dataset"
def read(p):return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
def csvout(n,rs):
 with (O/(n+".csv")).open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def save(f,n):f.tight_layout();f.savefig(O/(n+".pdf"));f.savefig(O/(n+".png"),dpi=250);plt.close(f)
M={r["id"]:r for r in read(D/"annotations/qa_master.jsonl")}
ans={tr:{(r["id"],r["model_key"]):r for r in read(R/f"results/scores/{tr}/answer/scores.jsonl")} for tr in ["direct","explain"]}
ev={(r["id"],r["model_key"]):r for r in read(R/"results/scores/explain/evidence/scores.jsonl")}
metrics=[]
for m in ["base","A","E"]:
 rs=[r for (sid,mm),r in ans["explain"].items() if mm==m and not r["task"].startswith("T14")];good=[r for r in rs if r["grade"]["rating"]==3];full=sum(ev[r["id"],m]["grade"]["rating"]==3 for r in good);bad=sum(ev[r["id"],m]["grade"]["rating"] in [1,2] for r in good);u=len(good)-full-bad
 metrics.append(dict(model=m,n=len(rs),answer_full=len(good),evidence_given_correct_lower=full/len(good),evidence_given_correct_upper=(full+u)/len(good),unsupported_correct_confirmed=bad/len(good),evidence_unresolved_given_correct=u/len(good),joint=full/len(rs)))
csvout("conditional_evidence",metrics)
# Both answers correct is stricter than mere output change; no lexical parsing.
pairs=[]
for name in ["visual_pairs","condition_pairs"]:
 pool=[p for p in read(R/"dataset/evaluation"/f"{name}.jsonl") if p["split"]=="test"]
 for tr in ans:
  for m in (["base","A","E","internvl","llava"] if tr=="direct" else ["base","A","E"]):
   for kind in sorted({p["kind"] for p in pool}):
    ps=[p for p in pool if p["kind"]==kind];full=upper=0
    for p in ps:
     rr=[ans[tr][sid,m]["grade"]["rating"] for sid in p["qa_ids"]];full+=all(x==3 for x in rr);upper+=all(x==3 or x is None for x in rr)
    pairs.append(dict(track=tr,model=m,kind=kind,n=len(ps),both_correct=full,lower=full/len(ps),upper=upper/len(ps)))
csvout("paired_changes",pairs)
# Reference graph complexity measured before model outputs, retaining every bin.
strata=[]
for feature in ["conditions","path_edges"]:
 for tr in ans:
  for m in ["A","E"]:
   groups=collections.defaultdict(list)
   for (sid,mm),r in ans[tr].items():
    if mm!=m or r["task"].startswith("T14"):continue
    checks=M[sid].get("verification",{}).get("graph_checks",[])
    if not checks:continue
    val=max(len(c.get("states",{})) if feature=="conditions" else len(c.get("path_edges",[])) for c in checks)
    label=("0" if val==0 else "1" if val==1 else "2" if val==2 else "3+") if feature=="conditions" else ("no recorded path" if val==0 else "1-3" if val<=3 else "4-6" if val<=6 else "7+")
    groups[label].append(r)
   for label,rs in groups.items():strata.append(dict(feature=feature,bin=label,track=tr,model=m,n=len(rs),accuracy=sum(r["grade"]["rating"]==3 for r in rs)/len(rs)))
csvout("complexity",strata)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,feature in zip(axes,["conditions","path_edges"]):
 bins=sorted({r["bin"] for r in strata if r["feature"]==feature})
 for m in ["A","E"]:
  rr={r["bin"]:r for r in strata if r["feature"]==feature and r["track"]=="explain" and r["model"]==m};ax.plot(bins,[rr[b]["accuracy"]*100 for b in bins],"o-",label=m)
 ax.set(title={"conditions":"Explicit contact states","path_edges":"Recorded path length"}[feature],xlabel="Reference complexity bin",ylabel="Answer full credit (%)");ax.legend();ax.tick_params(axis="x",rotation=20)
save(fig,"complexity")
fig,ax=plt.subplots(figsize=(7,4));x=np.arange(3);v=[r["evidence_given_correct_lower"]*100 for r in metrics];u=[r["evidence_given_correct_upper"]*100 for r in metrics];ax.bar(x,v,color=["#999","#0072b2","#009e73"]);ax.errorbar(x,v,yerr=[np.zeros(3),np.array(u)-v],fmt="none",c="black",capsize=5);ax.set(xticks=x,xticklabels=["Base","A","E"],ylim=(0,105),ylabel="Supported evidence among correct answers (%)",title="Conditional evidence validity; whiskers = unresolved bounds");save(fig,"conditional_evidence")
fig,ax=plt.subplots(figsize=(8,4));ks=sorted({r["kind"] for r in pairs})
for j,m in enumerate(["A","E"]):
 vals=[next(r["lower"]*100 for r in pairs if r["track"]=="explain" and r["model"]==m and r["kind"]==k) for k in ks];ax.bar(np.arange(len(ks))+(j-.5)*.35,vals,.35,label=m)
ax.set(xticks=range(len(ks)),xticklabels=[k.replace('_',' ') for k in ks],ylabel="Both members correct (%)",title="Controlled-change pairs · common explanation prompt");ax.tick_params(axis="x",rotation=12);ax.legend();save(fig,"paired_changes")
print(json.dumps(metrics,indent=2));print(json.dumps([r for r in pairs if r["track"]=="explain" and r["model"] in ["A","E"]],indent=2))

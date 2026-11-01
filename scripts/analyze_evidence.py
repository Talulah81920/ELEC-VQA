from pathlib import Path
import json,csv,collections,random,hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/"results/final_analysis";O.mkdir(exist_ok=True)
def read(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
def writecsv(name,rows):
 with (O/(name+".csv")).open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,name):
 fig.tight_layout();fig.savefig(O/(name+".png"),dpi=250);fig.savefig(O/(name+".pdf"));plt.close(fig)
master={r["id"]:r for r in read(R/"dataset/annotations/qa_master.jsonl")}
a=read(R/"results/scores/explain/answer/scores.jsonl");e=read(R/"results/scores/explain/evidence/scores.jsonl")
A={(r["id"],r["model_key"]):r for r in a};E={(r["id"],r["model_key"]):r for r in e};assert len(A)==len(E)==6222 and A.keys()==E.keys()
joined=[]
for k,x in A.items():
 y=E[k];assert x["prediction_sha256"]==y["prediction_sha256"]
 if x["task"].startswith("T14"):continue
 ar,er=x["grade"]["rating"],y["grade"]["rating"]
 # Joint is resolved failure if either dimension definitively lacks full credit.
 low=int(ar==3 and er==3);up=int((ar==3 or ar is None) and (er==3 or er is None))
 joined.append(dict(id=k[0],model=k[1],task=x["task"],drawing=x["drawing_id"],answer=ar,evidence=er,joint=low,joint_upper=up,origin=master[k[0]]["origin"],conditions=len(master[k[0]].get("verification",{}).get("graph_checks",[{}])[0].get("states",{})) if master[k[0]].get("verification",{}).get("graph_checks") else 0))
def stats(rs):
 n=len(rs);return dict(n=n,answer=sum(r["answer"]==3 for r in rs)/n,evidence=sum(r["evidence"]==3 for r in rs)/n,joint=sum(r["joint"] for r in rs)/n,joint_upper=sum(r["joint_upper"] for r in rs)/n,answer_unresolved=sum(r["answer"] is None for r in rs),evidence_unresolved=sum(r["evidence"] is None for r in rs))
summary=[dict(model=m,**stats([r for r in joined if r["model"]==m])) for m in ["base","A","E"]];writecsv("joint_overall",summary)
bytask=[dict(model=m,task=t,**stats([r for r in joined if r["model"]==m and r["task"]==t])) for t in sorted({r["task"] for r in joined}) for m in ["base","A","E"]];writecsv("joint_by_task",bytask)
strata=[]
for kind in ["origin","conditions"]:
 for val in sorted({r[kind] for r in joined},key=str):
  for m in ["A","E"]:
   rs=[r for r in joined if r["model"]==m and r[kind]==val];strata.append(dict(group=kind,value=val,model=m,**stats(rs)))
writecsv("strata",strata);writecsv("aligned_rows",joined)
boot=[]
for metric in ["answer","evidence","joint"]:
 pairs={m:{r["id"]:r for r in joined if r["model"]==m} for m in ["A","E"]};groups=collections.defaultdict(list)
 for sid,x in pairs["E"].items():
  y=pairs["A"][sid];v=(int(x[metric]==3)-int(y[metric]==3)) if metric!="joint" else x[metric]-y[metric];groups[x["drawing"]].append(v)
 rng=random.Random(42);ks=list(groups);bs=sorted(np.mean([v for k in rng.choices(ks,k=len(ks)) for v in groups[k]]) for _ in range(2000));boot.append(dict(metric=metric,E_minus_A=np.mean([v for vs in groups.values() for v in vs]),ci_low=bs[49],ci_high=bs[1949],clusters=len(ks)))
(O/"bootstrap.json").write_text(json.dumps(boot,indent=2),encoding="utf-8")
fig,ax=plt.subplots(figsize=(7,4));x=np.arange(3)
for j,k in enumerate(["answer","evidence","joint"]):
 bars=ax.bar(x+(j-1)*.24,[r[k]*100 for r in summary],.24,label=k);ax.bar_label(bars,fmt="%.1f",padding=2)
ax.set(xticks=x,xticklabels=["Base","A","E"],ylim=(0,85),ylabel="Full-credit rate (%)",title="Common explanation prompt · held-out test");ax.legend();save(fig,"answer_evidence_joint")
fig,ax=plt.subplots(figsize=(9,4));ts=sorted({r["task"] for r in joined})
for metric in ["answer","joint"]:
 vals=[]
 for t in ts:
  rr={r["model"]:r for r in bytask if r["task"]==t};vals.append((rr["E"][metric]-rr["A"][metric])*100)
 ax.plot(range(len(ts)),vals,"o-",label=metric)
ax.axhline(0,c="gray",ls="--");ax.set(xticks=range(len(ts)),xticklabels=[t[:3] for t in ts],ylabel="E minus A (percentage points)",title="Task-specific advantages and disadvantages");ax.legend();save(fig,"E_minus_A_tasks")
fig,ax=plt.subplots(figsize=(7,4));cats=[("answer",3,"evidence",3),("answer",3,"evidence",1),("answer",1,"evidence",3),("answer",1,"evidence",1)]
for m in ["base","A","E"]:
 rs=[r for r in joined if r["model"]==m];counts=collections.Counter(("full" if r["answer"]==3 else "nonfull" if r["answer"] is not None else "unresolved","full" if r["evidence"]==3 else "nonfull" if r["evidence"] is not None else "unresolved") for r in rs);(O/(m+"_contingency.json")).write_text(json.dumps({str(k):v for k,v in counts.items()},indent=2))
 bottom=0
 for key,color in [(('full','full'),'#009e73'),(('full','nonfull'),'#e69f00'),(('nonfull','full'),'#56b4e9'),(('nonfull','nonfull'),'#d55e00')]:
  v=counts[key]/len(rs)*100;ax.bar(m,v,bottom=bottom,color=color,label=str(key) if m=="base" else None);bottom+=v
 ax.bar(m,100-bottom,bottom=bottom,color='#bbb',label='At least one unresolved' if m=='base' else None)
ax.set(ylabel="Share of responses (%)",title="Answer / evidence full-credit combinations");ax.legend(fontsize=7,bbox_to_anchor=(1.02,1));save(fig,"joint_contingency")
lines=["# 最终自动评测结果与结论","","保留现有Qwen评分，无人工改分、无重评、无删题。主测试2064题，T14另列。答案和证据按相同prediction_sha256逐条对齐。未决保留，联合上界只将可能同时满分的未决视为成功。","","|模型|答案满分|证据满分|联合满分|联合上界|证据未决数|","|---|---:|---:|---:|---:|---:|"]
for r in summary:lines.append(f"|{r['model']}|{r['answer']*100:.2f}%|{r['evidence']*100:.2f}%|{r['joint']*100:.2f}%|{r['joint_upper']*100:.2f}%|{r['evidence_unresolved']}|")
lines += ["","## E在哪些条件下更好","","同一解释提示下，按任务和显式状态数量完整列出结果，见joint_by_task.csv与strata.csv；这些是预定义属性的探索性分组，不能挑选后宣称普遍优势。"]
for t in ts:
 rr={r['model']:r for r in bytask if r['task']==t};lines.append(f"- {t}: 答案E−A={(rr['E']['answer']-rr['A']['answer'])*100:+.2f}个百分点；联合E−A={(rr['E']['joint']-rr['A']['joint'])*100:+.2f}个百分点，n={rr['E']['n']}。")
lines += ["","## 解释边界","","证据有效性是相对参考事实的可核验说明质量，不证明隐藏思维忠实性。A证据未决较多，须呈现上下界。只有一个训练seed，7张测试来源图纸；bootstrap不覆盖裁判系统偏差。多任务问答和回路变体相互相关。模型族预处理不同，耗时不是纯架构比较。E不在所有答案任务上优于A，但联合评价可以揭示说明能力。无新增独立人工校准，不能声称专家金标准。"]
(O/"最终结果分析.md").write_text("\n".join(lines),encoding="utf-8");print(json.dumps(summary,indent=2));print(json.dumps(boot,indent=2))

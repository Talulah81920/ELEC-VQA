"""Recompute evidence figures from the separate provisional score overlay."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
for name in ['analyze_evidence.py','analyze_extended.py']:
 source=(R/'scripts'/name).read_text(encoding='utf-8')
 source=source.replace('results/scores/','results/assistant_review/scores/').replace('results/final_analysis','results/assistant_review/final_analysis').replace('results/deep_analysis','results/assistant_review/deep_analysis')
 source=source.replace('fig.tight_layout();fig.savefig','fig.text(.99,.005,"Provisional assistant-resolved scores · pending human review",ha="right",fontsize=6,color="gray");fig.tight_layout();fig.savefig').replace('def save(f,n):f.tight_layout();','def save(f,n):f.text(.99,.005,"Provisional assistant-resolved scores · pending human review",ha="right",fontsize=6,color="gray");f.tight_layout();')
 exec(compile(source,str(R/'scripts'/name),'exec'),{'__file__':str(R/'scripts'/name),'__name__':'__main__'})

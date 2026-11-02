import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('validation',type=Path);p.add_argument('--output',type=Path,default=Path('occupancy.png'));a=p.parse_args()
cases=json.loads(a.validation.read_text()); counts=np.array([c['counts'] for c in cases]);empty=counts==0
fig,axes=plt.subplots(3,1,figsize=(13,11))
axes[0].bar(range(counts.shape[1]),(counts/counts.sum(1,keepdims=True)).mean(0)*100);axes[0].set_ylabel('Mean patch share (%)')
axes[1].bar(range(counts.shape[1]),empty.mean(0)*100,color='tomato');axes[1].set_ylabel('Empty cases (%)');axes[1].set_ylim(0,100)
axes[2].imshow(empty,aspect='auto',interpolation='nearest',cmap='Reds',vmin=0,vmax=1);axes[2].set_ylabel('Case');axes[2].set_xlabel('Slot ID (red = empty)')
fig.tight_layout();fig.savefig(a.output,dpi=180)
print('Mean empty slots: %.2f; cases with empty slots: %d/%d'%(empty.sum(1).mean(),empty.any(1).sum(),len(cases)))

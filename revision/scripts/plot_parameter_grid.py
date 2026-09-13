"""Rebuild Supplementary Fig. S5 from the original paired parameter scan."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
z = np.load(root/'inputs/scan.npz', allow_pickle=False)
meta = json.loads(z['metadata'].item())
b = json.loads(z['metrics'].item()).index('b')
deltas = [.008, .015, .025, .04, .06, .09]
intensities = [0, .05, .10, .20, .40, .80]
gain = np.zeros((6,6)); area = np.zeros((6,6))
for i, decay in enumerate(deltas):
    ids = sorted([k for k,m in enumerate(meta) if m['condition']=='none' and m['config']['decay']==decay], key=lambda k:meta[k]['seed'])
    reference = z['trajectories'][ids,:,b]
    for j, intensity in enumerate(intensities[1:], start=1):
        ids = sorted([k for k,m in enumerate(meta) if m['condition']=='context' and m['config']['decay']==decay and m['config']['mediation_intensity']==intensity], key=lambda k:meta[k]['seed'])
        difference = z['trajectories'][ids,:,b] - reference
        gain[i,j] = difference[:,100].mean()
        area[i,j] = (np.trapezoid(difference[:,100:],axis=1)/400).mean()
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':10,
 'axes.labelsize':11,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42})
fig, axes = plt.subplots(1,2,figsize=(7.2,3.8))
for ax, values, title in zip(axes,[gain,area],['(a) Gain at withdrawal $D_b(0)$','(b) Mean post-withdrawal gain $A_{400}$']):
    im=ax.imshow(values,origin='lower',aspect='auto',cmap='viridis',vmin=0)
    ax.set_xticks(range(6),[str(x) for x in intensities]);ax.set_yticks(range(6),[str(x) for x in deltas])
    ax.set_xlabel('Mediation intensity $u$');ax.set_ylabel(r'Weight-decay fraction $\delta$');ax.set_title(title)
    for i in range(6):
        for j in range(6):
            ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=10,color='white' if values[i,j]<values.max()*.6 else '#17313A')
    fig.colorbar(im,ax=ax,fraction=.045,pad=.025)
fig.tight_layout()
fig.savefig(root/'latex/fig4_parameter_regions.pdf',bbox_inches='tight')
plt.close(fig)
np.savez_compressed(root/'results/parameter_grid.npz',gain=gain,area=area,decay=deltas,intensity=intensities)
print('Regenerated Supplementary Fig. S5 from 36 original grid cells.')

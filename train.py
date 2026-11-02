"""Portable full-bag training entrypoint; paths are supplied by a JSON manifest."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from resampler import SlotAttnResampler, OrthogonalPerceiver, ReconstructionPerceiver

class FullBags(Dataset):
    def __init__(self, records, base):
        if not records:
            raise ValueError('Each split must contain cases')
        self.records, self.base = records, base
    def __len__(self):
        return len(self.records)
    def __getitem__(self, i):
        paths = self.records[i]['paths']
        if not paths:
            raise ValueError('Case has no slides')
        x = torch.cat([torch.load(str(self.base / p), map_location='cpu') for p in paths]).float()
        if x.ndim != 2 or x.shape[0] == 0 or not torch.isfinite(x).all():
            raise ValueError('Expected finite, nonempty [N,D] feature tensors')
        return x, i

def build_model(args):
    kw = dict(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_slots=args.num_slots)
    if args.model == 'slot':
        return SlotAttnResampler(slot_iterations=args.iterations if args.iterations is not None else 4, **kw)
    cls = OrthogonalPerceiver if args.model == 'orth' else ReconstructionPerceiver
    return cls(iterations=args.iterations if args.iterations is not None else 1, **kw)

def objective(result, args):
    return sum(result[k] * weight for k, weight in
               [('loss_rec', args.rec_weight), ('loss_orth', args.orth_weight)] if k in result)

@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval(); cases = []
    for x, idx in loader:
        result = model(x.to(device), return_details=True)
        assignment = result['details']['assignment'][0]
        counts = torch.bincount(assignment.argmax(-1), minlength=args.num_slots).cpu()
        cases.append(dict(case_id=loader.dataset.records[int(idx)]['case_id'],
                          patches=x.shape[1], counts=counts.tolist(),
                          empty_slots=torch.where(counts == 0)[0].tolist(),
                          metrics={k: float(v) for k,v in result['monitor'].items()},
                          loss=float(objective(result,args))))
    key = 'query_orthogonality' if args.model == 'orth' else 'decoder_ratio'
    return float(np.mean([c['metrics'][key] for c in cases])), cases

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=['slot','orth','rec'],required=True)
    p.add_argument('--iterations',type=int)
    p.add_argument('--input-dim',type=int,default=1024)
    p.add_argument('--hidden-dim',type=int,default=256)
    p.add_argument('--num-slots',type=int,default=50)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--gc',type=int,default=8)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--orth-weight',type=float,default=1.)
    p.add_argument('--rec-weight',type=float,default=1.)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if a.gc<1 or a.epochs<1 or a.num_slots<2 or (a.iterations is not None and a.iterations<1):
        p.error('Require gc/epochs/iterations >= 1 and num-slots >= 2')
    torch.set_num_threads(4); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    manifest=json.loads(a.manifest.read_text()); base=a.manifest.resolve().parent
    ids={s:[r['case_id'] for r in manifest[s]] for s in ('train','val')}
    if any(len(v)!=len(set(v)) for v in ids.values()) or set(ids['train']) & set(ids['val']):
        raise ValueError('Case IDs must be unique and disjoint between train/val')
    loaders={s:DataLoader(FullBags(manifest[s],base),batch_size=1,shuffle=s=='train',
                         num_workers=a.num_workers) for s in ('train','val')}
    a.output.mkdir(parents=True,exist_ok=False)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    (a.output/'settings.json').write_text(json.dumps(settings,indent=2))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    model=build_model(a).to(a.device)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    best=float('inf'); history=[]
    for epoch in range(a.epochs+1):
        total=0.
        if epoch:
            model.train(); opt.zero_grad(set_to_none=True)
            loader=loaders['train']
            for step,(x,_) in enumerate(loader):
                loss=objective(model(x.to(a.device)),a)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                group=min(a.gc,len(loader)-(step//a.gc)*a.gc)
                total+=float(loss.detach()); (loss/group).backward()
                if (step+1)%a.gc==0 or step+1==len(loader):
                    opt.step(); opt.zero_grad(set_to_none=True)
        score,cases=evaluate(model,loaders['val'],a.device,a)
        row=dict(epoch=epoch,val_score=score,train_loss=total/len(loaders['train']) if epoch else None)
        history.append(row)
        checkpoint=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,settings=settings)
        torch.save(checkpoint,a.output/'last.pt')
        if score<best: best=score; torch.save(checkpoint,a.output/'best.pt')
        (a.output/('validation_epoch_%04d.json'%epoch)).write_text(json.dumps(cases,indent=2))
        (a.output/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)

if __name__=='__main__': main()

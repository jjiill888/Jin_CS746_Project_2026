"""Source-only teacher loss and fixed development selection."""
import torch
import torch.nn.functional as F

def source_kl(student, teacher, source, temperature=2.):
    if not bool(source.any()):
        return student.sum()*0
    return F.kl_div(F.log_softmax(student[source]/temperature,dim=-1),
                    F.softmax(teacher[source].detach()/temperature,dim=-1),
                    reduction='batchmean')*temperature**2

def choose(rows):
    baseline=next(r for r in rows if r['model']=='a000')
    order=['a000','last_layer','kd10','kd1','full_ce']
    for r in rows:
        r['feasible']=(r['source_selection_detection']>=baseline['source_selection_detection']-.01-1e-12
                       and r['source_selection_fa']<=baseline['source_selection_fa']+.001+1e-12)
    return min((r for r in rows if r['feasible']),key=lambda r:(r['l2_val_macro_fa'],order.index(r['model'])))

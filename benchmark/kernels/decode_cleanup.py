import statistics
import torch
import triton
from freetoken.kernel.fla.layernorm_gated import rms_norm_gated
import freetoken.kernel.triton.qsa.fused_decode as fused

def bench(fn):
    return statistics.median(triton.testing.do_bench_cudagraph(fn, rep=50)*1000 for _ in range(5))

torch.manual_seed(42)
for t in (1,3,4):
    x=torch.randn(t,6144,device='cuda',dtype=torch.bfloat16)
    z=torch.randn(t,16384,device='cuda',dtype=torch.bfloat16)[:,10240:]
    w=torch.randn(128,device='cuda',dtype=torch.bfloat16)
    def norm(grouped):
        return rms_norm_gated(x=x if grouped else x.reshape(-1,128),z=z if grouped else z.reshape(-1,128),weight=w,bias=None,group_size=128 if grouped else None,is_rms_norm=True,activation='sigmoid')
    before=bench(lambda:norm(False)); after=bench(lambda:norm(True))
    print(f'GDN T={t} old={before:.3f}us new={after:.3f}us speedup={before/after:.3f}',flush=True)
    q=torch.randn(t,24,256,device='cuda',dtype=torch.bfloat16); k=torch.randn(t,2,256,device='cuda',dtype=torch.bfloat16)
    qw=torch.randn(256,device='cuda',dtype=torch.bfloat16)*.1; kw=torch.randn_like(qw)*.1
    pos=torch.arange(12,12+t,device='cuda',dtype=torch.int32)
    inv=1/(10000**(torch.arange(32,device='cuda').float()/32)); angles=torch.outer(torch.arange(1024,device='cuda').float(),inv)
    cs=torch.cat((angles.cos(),angles.sin()),1).contiguous()
    fn=lambda:fused.qsa_qk_norm_rope_fused(q,k,pos,cs,qw,kw,1e-6)
    fused._PARALLEL_HEADS=False; before=bench(fn)
    fused._PARALLEL_HEADS=True; after=bench(fn)
    print(f'QSA norm T={t} old={before:.3f}us new={after:.3f}us speedup={before/after:.3f}',flush=True)
    v=torch.randn_like(k); kc=torch.zeros(512,2,256,device='cuda',dtype=torch.float8_e4m3fn); vc=torch.zeros_like(kc)
    ks=torch.zeros(512,2,device='cuda'); vs=torch.zeros_like(ks)
    raw=torch.randn(t,128,device='cuda',dtype=torch.bfloat16); iq=torch.randn(t,512,device='cuda',dtype=torch.bfloat16)
    qo=torch.empty(t,4,128,device='cuda',dtype=torch.bfloat16); ring=torch.randn(1,8,128,device='cuda',dtype=torch.bfloat16)
    cmp=torch.zeros(256,128,device='cuda',dtype=torch.bfloat16)
    ir=torch.zeros(t,device='cuda',dtype=torch.int32); slots=torch.zeros(1,device='cuda',dtype=torch.int32)
    cu=torch.tensor([0,t],device='cuda',dtype=torch.int32)
    cmp_rows=torch.where(pos%4==3,pos//4,255).int(); ring_rows=(pos%8).int()
    iw=torch.randn(128,device='cuda',dtype=torch.bfloat16)*.1
    fn=lambda:fused.qsa_index_write_fused(k=k.view(t,-1),v=v.view(t,-1),out_loc=pos,k_cache=kc,v_cache=vc,k_scale=ks,v_scale=vs,raw_k=raw,idx_q=iq,q_out=qo,ring=ring,ring_slots=slots,token_to_req=ir,cu_seqlens=cu,positions=pos,cmp_cache=cmp,cmp_rows=cmp_rows,ring_rows=ring_rows,cos_sin=cs,k_norm_weight=iw,q_norm_weight=iw,eps=1e-6,ratio=4)
    fused._DIRECT_PERMUTE=False; before=bench(fn)
    fused._DIRECT_PERMUTE=True; after=bench(fn)
    print(f'QSA index T={t} old={before:.3f}us new={after:.3f}us speedup={before/after:.3f}',flush=True)

"""Full-patch resampling models extracted from the existing experiments.
Input: [B,N,D]; no patch sampling. Original state_dict keys are preserved.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

class CrossAttn(nn.Module):
    def __init__(self,input_dim,hidden_dim=256,head=None):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.head=head
        self.q_proj=nn.Linear(input_dim,hidden_dim)
        self.k_proj=nn.Linear(input_dim,hidden_dim)
        self.v_proj=nn.Linear(input_dim,hidden_dim)
        self.out_proj=nn.Linear(hidden_dim,hidden_dim)
    def forward(self,Q,K,V,return_logits=False):
        Q=self.q_proj(Q)
        K=self.k_proj(K)
        V=self.v_proj(V)
        if self.head is not None:
            B,N,M=Q.shape[0],Q.shape[1],K.shape[1]
            head_dim=self.hidden_dim//self.head
            Q=Q.reshape(B,N,self.head,-1).transpose(1,2)
            K=K.reshape(B,M,self.head,-1).transpose(1,2)
            V=V.reshape(B,M,self.head,-1).transpose(1,2)
            A=torch.bmm(Q, K.transpose(-2,-1)) / math.sqrt(head_dim)
            # A: (B,heads,N,M)
            weights = F.softmax(A,dim=-1)
            seq=torch.bmm(weights, V).transpose(1,2).reshape(B,N,-1) # (B,heads,N,head_dim)->(B,N,heads,head_dim)->(B,N,D)
            out = self.out_proj(seq) # (B,N,D)
            return out
        else:
            A = torch.bmm(Q, K.transpose(-2,-1)) / math.sqrt(self.hidden_dim)
            # A: (B,N,M)
            weights = F.softmax(A, dim=-1)
            out = self.out_proj(torch.bmm(weights, V)) #(B,N,D)
            if return_logits:
                return out,A
            return out

class PerceiverTransLayer(nn.Module):
    def __init__(self, dim=256,heads=8,expand=2):
        super(PerceiverTransLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim,elementwise_affine=False,bias=False)
        self.norm2 = nn.LayerNorm(dim,elementwise_affine=False,bias=False)
        self.attn = nn.MultiheadAttention(dim,num_heads=heads,dropout=0.0,batch_first=True)
        # self.ffn=nn.Sequential(nn.Linear(dim,dim*expand),nn.SiLU(),nn.Linear(dim*expand,dim),nn.Dropout(0.1))
    def forward(self, x):
        seq=self.norm1(x)
        # print('[PerceiverTransLayer norm1]', seq.max(),seq.min())
        attn_out = x + self.attn(seq, seq, seq,
        need_weights=False)[0]
        # print('[PerceiverTransLayer attn_out]', attn_out.max(),attn_out.min())
        # ffn_out = self.ffn(self.norm2(attn_out))
        # print('[PerceiverTransLayer ffn_out]', ffn_out.max(),ffn_out.min())
        # output = attn_out + ffn_out
        return attn_out

class Perceiver(nn.Module):
    def __init__(self, input_dim,latent_num=128,hidden_dim=256,trans_heads=8,trans_depth=6):
        super(Perceiver, self).__init__()
        # crossattn+latent block,交替进行
        self.latent=nn.Parameter(torch.randn(latent_num,hidden_dim),requires_grad=True)
        torch.nn.init.trunc_normal_(self.latent,mean=0,std=0.5)
        self.cross_attn =  CrossAttn(input_dim,hidden_dim)
        self.latent_trans=nn.Sequential(*[PerceiverTransLayer(dim=hidden_dim,heads=trans_heads) for _ in range(trans_depth)])
    def forward(self, x,iter_num=None):
        # x:(B,N,D)
        B,N,D=x.shape
        if iter_num is None:
            # print('[Perceiver input]', x.max().item(),x.min().item())
            latent=self.latent.unsqueeze(0).repeat(B,1,1)
            # print('[Perceiver latent]', latent.max().item(),latent.min().item())
            cross_output= self.cross_attn(latent, x, x)
            # print('[Perceiver cross_output]', cross_output.max().item(),cross_output.min().item())
            latent_output=self.latent_trans(cross_output)
            # print('[Perceiver latent_output]', latent_output.max().item(),latent_output.min().item())
            return latent_output
        else:
            assert iter_num>=1
            latent=self.latent.unsqueeze(0).repeat(B,1,1)
            for _ in range(iter_num):
                cross_output= self.cross_attn(latent, x, x)
                latent=self.latent_trans(cross_output)
            return latent

def normalize_prototypes(x, method='l2', eps=1e-8):
    """Normalize [N,D] or [B,N,D] features within each bag, across tokens."""
    if method == 'raw':
        return x
    if method == 'l2':
        return F.normalize(x, p=2, dim=-1, eps=eps)
    if method == 'zscore':
        # Match the Resampler experiment's sample std; handle singleton bags.
        correction = 1 if x.shape[-2] > 1 else 0
        return (x - x.mean(-2, keepdim=True)) / (x.std(-2, keepdim=True, unbiased=bool(correction)) + eps)
    if method == 'minmax':
        lo, hi = x.amin(-2, keepdim=True), x.amax(-2, keepdim=True)
        return (x - lo) / (hi - lo + eps)
    if method == 'legacy':
        return x - x.amin(-1, keepdim=True)
    raise ValueError('Unknown normalization: {}'.format(method))

class KmeansResampler(nn.Module):
    def __init__(self,num_prototypes=256,iters=10):
        super().__init__()
        self.num_prototypes=num_prototypes
        self.iters=iters
    @torch.no_grad()
    def _normalization(self,x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    @torch.no_grad()
    def forward(self,x):
        # x(B,N,D)-->(B,Q,D)
        B,N,D=x.shape
        if N == 0 or self.num_prototypes < 1:
            raise ValueError('Kmeans needs nonempty input and positive num_prototypes')
        device=x.device
        out=[]
        for b in range(B):
            x_b=self._normalization(x[b])#(N,D)
            # 初始化原型
            first_idx=torch.randint(0,N,(1,),device=device)
            centers = [x_b[first_idx]]           # (1, D)
            # print(centers[0].shape)
            for _ in range(self.num_prototypes - 1):
                # print(torch.cat(centers,dim=0).shape)
                # print(x_b.shape)
                dist = torch.cdist(x_b, torch.cat(centers,dim=0)).min(dim=1).values ** 2
                centers.append(x_b[dist.argmax()].unsqueeze(0))
            C = torch.cat(centers)
            for _ in range(self.iters):
                assign=torch.cdist(x_b,C).argmin(dim=1)#(N,)
                C_new=torch.zeros_like(C)
                cnt=torch.zeros(self.num_prototypes,device=device)
                C_new.index_add_(0,assign,x_b)
                cnt.index_add_(0,assign,torch.ones(N,device=device))
                # index_add_(dim, index, source) 的语义：对每一行 i，执行 C_new[assign[i]] += xb[i]。
                empty = cnt < 1
                k=int(empty.sum().item())
                if empty.any():
                    dmin = torch.cdist(x_b, C).min(dim=1).values
                    far_idx=dmin.argsort(descending=True).repeat((k + N - 1) // N)[:k]
                    C_new[empty] = x_b[far_idx]
                    cnt[empty] = 1
                C=C_new/cnt.unsqueeze(-1)
            out.append(C)
        return torch.stack(out)

class SlotAttnResampler(nn.Module):
    def __init__(self,input_dim=1024,hidden_dim=256,num_slots=50,slot_iterations=4):
        super().__init__()
        self.slot_iterations = slot_iterations
        self.num_slots=num_slots
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.slot = nn.Parameter(torch.randn(1, num_slots, hidden_dim) * .2)

        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.dot_factor=math.sqrt(hidden_dim)
        self.decoder = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, input_dim))
    def _compute_assignment(self,tokens,slots,mode,temperature=1.):
        k=self.proj_k(tokens)
        q=self.proj_q(self.slot_norm(slots))
        logits=torch.einsum('bnd,bkd->bnk',k,q)/(self.dot_factor*temperature)
        if mode == 'cross':
            assignment=logits.softmax(-2)
            weight=assignment
        elif mode == 'compete':
            assignment=logits.softmax(-1)
            weight=assignment / assignment.sum(1, keepdim=True).clamp_min(1e-8)
        return assignment,weight
    def forward(self,x,temperature=1.,return_details=False,return_slots=False):
        # x:(B, N, D)
        # 归一化+encoder
        B,N,K=x.shape[0],x.shape[1],self.num_slots
        x=F.normalize(x,dim=-1)
        mean=x.mean(1,keepdim=True)
        scale = (x - mean).square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        x_target= (x - mean) / scale
        x_token=self.encoder(x_target)
        # 初始化slot：
        slot=self.slot.expand(B,-1,-1)
        # cross attention update--->n方向softmax,matmul v--->init
        slot=slot+self._compute_assignment(x_token,slot,'cross',temperature)[1].transpose(-1,-2) @ self.proj_v(x_token)
        for _ in range(self.slot_iterations):
            # compete attention update--->k方向softmax,normalize-->matmul v--->update
            update=self._compute_assignment(x_token,slot,'compete',temperature)[1].transpose(-1,-2) @ self.proj_v(x_token)
            # gate update
            gate = self.gate(torch.cat((slot, update), dim=-1))
            slot = (1-gate)*slot + gate*update
        # decode
        assignment,weight=self._compute_assignment(x_token,slot,'compete',temperature)
        slots_decoded=self.decoder(slot)
        reconstruct=assignment @ slots_decoded
        loss=(reconstruct-x_target).square().mean()
        # 用同一分配聚合原始特征，得到实际进入GW的重心
        prototypes=weight.transpose(-1,-2) @ x
        mass=assignment.mean(1)
        # 监控不参与反向传播，每个病例只保留标量
        with torch.no_grad():
            baseline=(x-mean).square().mean((1,2)).clamp_min(1e-12)
            monitor=dict(decoder_ratio=(reconstruct-x_target).square().mean((1,2)),
                         decoder_mse=(mean+scale*reconstruct-x).square().mean((1,2)),
                         barycentric_ratio=(assignment @ prototypes-x).square().mean((1,2))/baseline,
                         assignment_entropy=-(assignment*assignment.clamp_min(1e-30).log()).sum(-1).mean(1)/math.log(max(K,2)),
                         slots_mass_effective=torch.exp(-(mass*mass.clamp_min(1e-30).log()).sum(-1)))
        result=dict(prototypes=prototypes,mass=mass,loss_rec=loss,monitor=monitor)
        if return_slots:
            # 下游任务直接使用hidden_dim状态，保留生存损失的梯度。
            result['slots']=slot
        if return_details:
            # 仅画图时返回大张量，不保留其计算图
            result['details']=dict(slots=slot.detach(),assignment=assignment.detach())
        return result
    def forward_details(self,x,temperature=1.):
        return self.forward(x,temperature,return_details=True)

class OrthogonalPerceiver(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_slots=50,
                 self_attn_layers=1, iterations=4, heads=8, orthogonal_init=True):
        super().__init__()
        if iterations < 1 or self_attn_layers < 1 or num_slots > hidden_dim:
            raise ValueError('Require positive iterations/depth and num_slots <= hidden_dim')
        # The additional H->H input adapter makes the default trainable parameter
        # count exactly equal to SlotAttnResampler, with no unused decoder.
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                     nn.Linear(hidden_dim, hidden_dim))
        self.perceiver = Perceiver(hidden_dim, latent_num=num_slots, hidden_dim=hidden_dim,
                                   trans_heads=heads, trans_depth=self_attn_layers)
        if orthogonal_init:
            nn.init.orthogonal_(self.perceiver.latent)
        self.iterations = iterations
        self.num_slots = num_slots

    @staticmethod
    def orthogonality(slots):
        q = F.normalize(slots, dim=-1).transpose(-1, -2)
        gram = q.transpose(-1, -2) @ q
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        return torch.linalg.matrix_norm(gram - eye, ord='fro', dim=(-2, -1))

    def forward(self, x, return_slots=False, return_details=False):
        x = F.normalize(x, dim=-1)
        mean = x.mean(1, keepdim=True)
        scale = (x - mean).square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        tokens = self.encoder((x - mean) / scale)
        slots = self.perceiver(tokens, iter_num=self.iterations)
        orth = self.orthogonality(slots)
        # Read out from the FINAL refined queries through the existing cross
        # attention projections. This readout is diagnostic only, not an update.
        with torch.no_grad():
            _, logits = self.perceiver.cross_attn(slots, tokens, tokens, return_logits=True)
            assignment = logits.transpose(-1, -2).softmax(-1)
            weight = assignment / assignment.sum(1, keepdim=True).clamp_min(1e-8)
            prototypes = weight.transpose(-1, -2) @ x
            mass = assignment.mean(1)
            baseline = (x - mean).square().mean((1, 2)).clamp_min(1e-12)
            counts = F.one_hot(assignment.argmax(-1), self.num_slots).sum(1)
            monitor = dict(
                query_orthogonality=orth.detach(),
                learned_query_orthogonality=self.orthogonality(self.perceiver.latent[None]).expand(x.shape[0]),
                barycentric_ratio=(assignment @ prototypes - x).square().mean((1, 2)) / baseline,
                assignment_entropy=-(assignment * assignment.clamp_min(1e-30).log()).sum(-1).mean(1)
                                   / math.log(self.num_slots),
                slots_mass_effective=torch.exp(-(mass * mass.clamp_min(1e-30).log()).sum(-1)),
                occupied_slots=(counts > 0).sum(-1).float(),
                max_hard_fraction=counts.amax(-1).float() / x.shape[1])
        result = dict(loss_orth=orth.mean(), monitor=monitor, prototypes=prototypes, mass=mass)
        if return_slots:
            result['slots'] = slots
        if return_details:
            result['details'] = dict(slots=slots.detach(), assignment=assignment.detach())
        return result

class ReconstructionPerceiver(OrthogonalPerceiver):
    def __init__(self, decoder_type='independent', **kwargs):
        super().__init__(**kwargs)
        if decoder_type not in ('tied', 'independent'):
            raise ValueError('decoder_type must be tied or independent')
        self.decoder_type = decoder_type
        if decoder_type == 'independent':
            # Preserve the original frontend initialization and training sampler
            # RNG state, so adding a decoder does not change the case order.
            hidden_dim = self.perceiver.latent.shape[-1]
            input_dim = self.encoder[0].in_features
            with torch.random.fork_rng(devices=[]):
                self.decoder = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, input_dim))

    def decode(self, slots):
        if self.decoder_type == 'independent':
            return self.decoder(slots)
        # Reuse the frontend LayerNorm and transposed input projection. No
        # dormant parameters or extra trainable budget are added.
        return F.linear(self.encoder[1](slots), self.encoder[0].weight.transpose(0, 1))

    def forward(self, x, return_slots=False, return_details=False):
        x = F.normalize(x, dim=-1)
        mean = x.mean(1, keepdim=True)
        scale = (x - mean).square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        target = (x - mean) / scale
        tokens = self.encoder(target)
        slots = self.perceiver(tokens, iter_num=self.iterations)
        orth = self.orthogonality(slots)
        # This final-query readout now participates in reconstruction gradients.
        _, logits = self.perceiver.cross_attn(slots, tokens, tokens, return_logits=True)
        assignment = logits.transpose(-1, -2).softmax(-1)
        reconstruction = assignment @ self.decode(slots)
        rec = (reconstruction - target).square().mean((1, 2))
        with torch.no_grad():
            weight = assignment / assignment.sum(1, keepdim=True).clamp_min(1e-8)
            prototypes = weight.transpose(-1, -2) @ x
            mass = assignment.mean(1)
            baseline = (x - mean).square().mean((1, 2)).clamp_min(1e-12)
            counts = F.one_hot(assignment.argmax(-1), self.num_slots).sum(1)
            monitor = dict(
                query_orthogonality=orth.detach(),
                learned_query_orthogonality=self.orthogonality(self.perceiver.latent[None]).expand(x.shape[0]),
                decoder_ratio=rec.detach(),
                decoder_mse=(mean + scale * reconstruction - x).square().mean((1, 2)),
                barycentric_ratio=(assignment @ prototypes - x).square().mean((1, 2)) / baseline,
                assignment_entropy=-(assignment * assignment.clamp_min(1e-30).log()).sum(-1).mean(1)
                                   / math.log(self.num_slots),
                slots_mass_effective=torch.exp(-(mass * mass.clamp_min(1e-30).log()).sum(-1)),
                occupied_slots=(counts > 0).sum(-1).float(),
                max_hard_fraction=counts.amax(-1).float() / x.shape[1])
        result = dict(loss_orth=orth.mean(), loss_rec=rec.mean(), monitor=monitor,
                      prototypes=prototypes, mass=mass)
        if return_slots:
            result['slots'] = slots
        if return_details:
            result['details'] = dict(slots=slots.detach(), assignment=assignment.detach())
        return result

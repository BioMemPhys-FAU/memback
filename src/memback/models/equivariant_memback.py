import math
import torch
import torch.nn.functional as F
from torch import nn

class MartiniBeadEmbedding(nn.Module):
    def __init__(self, n_size=4, n_class=8, d_size=3, d_class=4, dropout=0.0):
        super().__init__()
        self.size_emb = nn.Embedding(n_size, d_size)
        self.class_emb = nn.Embedding(n_class, d_class)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = d_size + d_class
        nn.init.normal_(self.size_emb.weight, std=0.02)
        nn.init.normal_(self.class_emb.weight, std=0.02)

    def forward(self, bead_features: torch.LongTensor) -> torch.Tensor:
        class_id = bead_features[:, 0]
        size_id = bead_features[:, 1]
        emb = torch.cat([self.size_emb(size_id), self.class_emb(class_id)], dim=-1)
        return self.dropout(emb)

class GaussianRBF(nn.Module):
    def __init__(self, n_rbf=20, cutoff=12.0):
        super().__init__()
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, n_rbf)
        widths = torch.full((n_rbf,), (centers[1] - centers[0]).item())
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

    def forward(self, d):
        x = (d[:, None] - self.centers[None, :]) / self.widths[None, :]
        rbf = torch.exp(-0.5 * x * x)
        env = 0.5 * (torch.cos(math.pi * torch.clamp(d / self.cutoff, max=1.0)) + 1.0)
        return rbf * env[:, None]

class VectorLinear(nn.Module):
    def __init__(self, f_in, f_out):
        super().__init__()
        self.w = nn.Parameter(torch.empty(f_out, f_in))
        nn.init.xavier_uniform_(self.w)

    def forward(self, v):                        # [N, F_in, 3]
        return torch.einsum("gf,nfd->ngd", self.w, v)

class PaiNNMessage(nn.Module):
    def __init__(self, hidden, n_rbf):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4 * hidden)
        )
        self.filter_net = nn.Linear(n_rbf, 4 * hidden)
        self.hidden = hidden

    def forward(self, s, v, edge_index, r_hat, rbf):
        src, dst = edge_index[0], edge_index[1]
        phi = self.phi(s)[src]
        W = self.filter_net(rbf)
        x = phi * W
        ds, dv_s, dv_r, dv_c = torch.split(x, self.hidden, dim=-1)

        r_hat_e = r_hat[:, None, :]
        cross = torch.linalg.cross(v[src], r_hat_e.expand_as(v[src]), dim=-1)
        dv = (dv_s[..., None] * v[src]
              + dv_r[..., None] * r_hat_e
              + dv_c[..., None] * cross)

        s_agg = torch.zeros_like(s).index_add_(0, dst, ds)
        v_agg = torch.zeros_like(v).index_add_(0, dst, dv)
        return s + s_agg, v + v_agg

class PaiNNUpdate(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.U = VectorLinear(hidden, hidden)
        self.V = VectorLinear(hidden, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3 * hidden)
        )
        self.hidden = hidden

    def forward(self, s, v):
        Uv = self.U(v)
        Vv = self.V(v)
        Vv_norm = torch.linalg.norm(Vv, dim=-1)
        a = self.mlp(torch.cat([s, Vv_norm], dim=-1))
        a_vv, a_sv, a_ss = torch.split(a, self.hidden, dim=-1)

        dv = a_vv[..., None] * Uv
        inner = torch.sum(Uv * Vv, dim=-1)
        ds = a_ss + a_sv * inner
        return s + ds, v + dv

class EquivariantReadout(nn.Module):
    def __init__(self, hidden, max_atom):
        super().__init__()
        self.vec = VectorLinear(hidden, max_atom)
        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, max_atom)
        )

    def forward(self, s, v):
        out = self.vec(v)
        g = self.gate(s)
        return out * g[..., None]

class EquivariantBackmap(nn.Module):
    def __init__(self, max_atom_number, hidden_channels=384, n_layers=3,
                 dropout=0.04, n_rbf=20, cutoff=12.0, n_scalar_extra=3, pos_in_v=False, aa_emb_in=False,):
        super().__init__()
        self.emb = MartiniBeadEmbedding(dropout=dropout)
        in_scalar = self.emb.output_dim + n_scalar_extra
        self.scalar_in = nn.Linear(in_scalar, hidden_channels)
        if pos_in_v:
            self.vector_in = VectorLinear(1, hidden_channels)
        self.rbf = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

        self.messages = nn.ModuleList(
            PaiNNMessage(hidden_channels, n_rbf) for _ in range(n_layers)
        )
        self.updates = nn.ModuleList(
            PaiNNUpdate(hidden_channels) for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_channels) for _ in range(n_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = EquivariantReadout(hidden_channels, max_atom_number)
        self.hidden = hidden_channels

    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index
        emb = self.emb(x[:, :2].long())
        s = torch.cat([emb, x[:, 2:5].to(emb.dtype)], dim=-1)
        s = self.scalar_in(s) # [N, H]
        v = torch.zeros(s.shape[0], self.hidden, 3, device=s.device, dtype=s.dtype)
        src, dst = edge_index[0], edge_index[1]
        r = pos[dst] - pos[src]
        d = torch.linalg.norm(r, dim=-1)
        r_hat = r / (d[:, None] + 1e-8)
        rbf = self.rbf(d)

        for msg, upd, norm in zip(self.messages, self.updates, self.norms):
            s, v = msg(s, v, edge_index, r_hat, rbf)
            s, v = upd(s, v)
            s = self.dropout(norm(s))
        return self.readout(s, v)

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = dict(ckpt["config"])
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.to(map_location)
        model.eval()
        return model
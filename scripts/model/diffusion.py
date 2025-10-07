import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import math, random

# ---------- helpers ----------
def sinusoidal_embedding(t, dim=32):
    device, half = t.device, dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half-1))
    emb = t[:, None] * freqs[None]          # [B, half]
    return torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, dim]

def linear_beta_schedule(T, b0=1e-4, b1=0.02):
    return torch.linspace(b0, b1, T+1)      # include β_T

# ---------- tiny U-Net with text conditioning ----------
class Unet(nn.Module):
    def __init__(self, text_dim, hidden=128):
        super().__init__()
        self.inc  = nn.Conv2d(3, hidden, 3, padding=1)
        self.down = nn.Conv2d(hidden, hidden, 4, 2, 1)
        self.up   = nn.ConvTranspose2d(hidden*2, hidden, 4, 2, 1)
        self.out  = nn.Conv2d(hidden*2, 3, 3, padding=1)
        self.temb = nn.Linear(32, hidden)
        self.txt  = nn.Linear(text_dim, hidden)

    def forward(self, x, t_emb, txt_emb):
        t = self.temb(t_emb)[:, :, None, None]
        c = self.txt(txt_emb)[:, :, None, None]

        h1 = F.silu(self.inc(x) + t + c)
        h2 = F.silu(self.down(h1))

        h  = F.silu(self.up(torch.cat([h2, h2], dim=1)))   # skip conn
        h  = torch.cat([h, h1], 1)
        return self.out(h)

# ---------- training setup ----------
T = 1000
betas = linear_beta_schedule(T).to("cuda")
alphas = 1 - betas
alphas_cum = torch.cumprod(alphas, 0)

def q_sample(x0, t, noise):
    sqrt_ac = alphas_cum[t] ** 0.5
    sqrt_om = (1 - alphas_cum[t]) ** 0.5
    return sqrt_ac[:, None, None, None] * x0 + sqrt_om[:, None, None, None] * noise

# toy dataset: CIFAR-10 (25k samples)
tfm = transforms.Compose([transforms.Resize(32),
                          transforms.ToTensor(),
                          lambda x: x*2 - 1])
loader = DataLoader(datasets.CIFAR10(root='.', download=True, transform=tfm),
                    batch_size=64, shuffle=True, num_workers=2)

# model = TinyUNet(text_dim=512).cuda()
# opt = torch.optim.AdamW(model.parameters(), 1e-4)
# text_embed = torch.randn(len(loader.dataset), 512, device='cuda')  # fake text!
# print(text_embed.shape)

# for epoch in range(5):
#     for i,(img,_) in enumerate(loader):
#         img = img.cuda()
#         b = img.size(0)
#         print('img.shape: ',img.shape)

#         t = torch.randint(0, T, (b,), device='cuda')
#         print('t.shape: ',t.shape)

#         noise = torch.randn_like(img)
#         x_t = q_sample(img, t, noise)

#         out = model(x_t, sinusoidal_embedding(t.float()), text_embed[i*b:(i+1)*b])
#         loss = F.mse_loss(out, noise)
#         opt.zero_grad(); loss.backward(); opt.step()
#     print(f"epoch {epoch}: {loss.item():.4f}")

# # ---------- sampling (DDIM) ----------
# @torch.no_grad()
# def p_sample(x, t, t_next, txt_emb, eta=0.0):
#     eps = model(x, sinusoidal_embedding(t.float()), txt_emb)
#     alpha, alpha_next = alphas_cum[t], alphas_cum[t_next]
#     beta = 1 - alpha
#     x0_pred = (x - (beta**0.5)*eps) / (alpha**0.5)
#     sigma = eta * ((1 - alpha/alpha_next)*(1 - alpha_next)/ (1 - alpha))**0.5
#     noise = torch.randn_like(x) if eta else 0
#     return (alpha_next**0.5)*x0_pred + ((1 - alpha_next - sigma**2)**0.5)*eps + sigma*noise

# def sample(txt_emb, steps=50):
#     x = torch.randn(1,3,32,32, device='cuda')
#     ts = torch.linspace(T-1, 0, steps, dtype=torch.long, device='cuda')
#     for i in range(steps-1):
#         x = p_sample(x, ts[i], ts[i+1], txt_emb, eta=0.0)
#     return (x.clamp(-1,1)+1)/2     # to [0,1]

# img = sample(text_embed[0:1])



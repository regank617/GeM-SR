import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

try:
    import thop
except ImportError:
    thop = None

def _macs_to_gflops(macs: float) -> float:
    return (float(macs) * 2.0) / 1e9

def _var_lvl_pos_start_for_stage(var, si: int) -> int:
    return sum(int(var.patch_nums[j]) ** 2 for j in range(si))

def _varsr_lvl_pos_start_for_stage(v, si: int) -> int:
    if si <= 0:
        return 0
    c = int(getattr(v, "context_token", v.first_l))
    for j in range(1, si):
        c += int(v.patch_nums[j]) ** 2
    return c

def _varsr_freqs_slice(v, si: int, pn: int, cur_freq: int) -> Tuple[torch.Tensor, int]:
    if si == 0:
        ct = int(getattr(v, "context_token", v.first_l))
        return v.freqs_cis[0:ct], cur_freq + ct
    seg = int(pn) ** 2
    return v.freqs_cis[cur_freq : cur_freq + seg], cur_freq + seg


class _VAROneStage(nn.Module):
    def __init__(self, var):
        super().__init__()
        self.var = var

    def forward(self, next_token_map: torch.Tensor, cond_BD: torch.Tensor) -> torch.Tensor:
        cond_or_gss = self.var.shared_ada_lin(cond_BD)
        x = next_token_map
        for b in self.var.blocks:
            x = b(x=x, cond_BD=cond_or_gss, attn_bias=None)
        return self.var.get_logits(x, cond_BD)


class _VARSROneStage(nn.Module):
    def __init__(self, var):
        super().__init__()
        self.var = var

    def forward(
        self,
        next_token_map: torch.Tensor,
        cond_BD: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        cond_or_gss = self.var.shared_ada_lin(cond_BD)
        x = next_token_map
        for b in self.var.blocks:
            x = b(x=x, cond_BD=cond_or_gss, freqs_cis=freqs_cis, attn_bias=None)
        return self.var.get_logits(x, cond_BD)


class _ConEmbeddingFeat(nn.Module):
    def __init__(self, con_embedding):
        super().__init__()
        self.con_embedding = con_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.con_embedding(x)
        return out[0] if isinstance(out, (list, tuple)) else out


class _PredHeadPair(nn.Module):
    """ConvNeXt-style pred + head block (used for GeMSR thop profiling)."""

    def __init__(self, pred, head):
        super().__init__()
        self.pred = pred
        self.head = head

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.head(self.pred(hidden, cond))


class GeMSRWrapper(nn.Module):
    """
    End-to-end LR→SR for the hierarchical SSM (``build_model_v5``) + frozen conditioner.

    Sampling uses ``ssm.generate`` with configurable per-scale temperatures and optional
    refiner strength on tokenizer decode.
    """

    _DEFAULT_TEMPS = {"root": 1.0, "scale_0": 1.0, "scale_1": 1.0, "scale_2": 1.0}
    _GEMSR_TEMP_KEYS = ("root", "scale_0", "scale_1", "scale_2")

    def __init__(
        self,
        ssm,
        conditioner,
        cfg_scale: float = 1.0,
        *,
        refiner_strength: float = 1.0,
        temps: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.ssm, self.conditioner = ssm, conditioner
        self.cfg_scale = float(cfg_scale)
        self.refiner_strength = float(refiner_strength)
        if temps is None:
            self._temps = dict(type(self)._DEFAULT_TEMPS)
        else:
            self._temps = {k: float(temps[k]) for k in self._GEMSR_TEMP_KEYS}
        self.eval_result_meta = {
            "gemsr_eval_refiner_strength": self.refiner_strength,
            "gemsr_eval_temps": dict(self._temps),
        }

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        cond_outputs = self.conditioner(lr)
        tok = self.ssm.tokenizer
        output = self.ssm.generate(
            cond_outputs,
            tok,
            top_k=100,
            top_p=0.85,
            temps=dict(self._temps),
            cfg_scale=self.cfg_scale,
        )
        root = tok.decode_root(output["root"])
        s0 = tok.inverse_step(root, output["scale_0"], scale_idx=0)
        s1 = tok.inverse_step(s0, output["scale_1"], scale_idx=1)
        s2 = tok.inverse_step(s1, output["scale_2"], scale_idx=2)
        return tok.features_to_pixels(s2, refiner_strength=self.refiner_strength)

    def get_gflops(self, dummy_lr: torch.Tensor, device: torch.device) -> float:
        if thop is None:
            return 0.0
        self.eval()
        ssm, cond = self.ssm, self.conditioner
        d = dummy_lr.to(device)
        B = d.size(0)
        total_macs = 0.0
        try:
            macs_c, _ = thop.profile(cond, inputs=(d,), verbose=False)
            total_macs += float(macs_c)
            with torch.no_grad():
                co = cond(d)
                gs = ssm.global_proj(co["global_style"])
                cond_root = co["root_ll"].flatten(2).transpose(1, 2)
                cond_s0 = co["scale_0_packed"].flatten(2).transpose(1, 2)
                cond_s1 = co["scale_1_packed"].flatten(2).transpose(1, 2)
                cond_s2 = co["scale_2_packed"].flatten(2).transpose(1, 2)
                root_hidden = gs.unsqueeze(1).expand(B, 36, -1)
                pr = _PredHeadPair(ssm.pred_root, ssm.head_root).to(device).eval()
                macs_r, _ = thop.profile(pr, inputs=(root_hidden, cond_root), verbose=False)
                total_macs += float(macs_r)
                root_h = ssm.pred_root(root_hidden, cond_root)
                root_logits = ssm.head_root(root_h)
                tok_r = root_logits.argmax(dim=-1)
                curr_ll = ssm.tokenizer.decode_root(tok_r)
                ph0 = _PredHeadPair(ssm.pred_s0, ssm.head_s0).to(device).eval()
                h0 = curr_ll.flatten(2).transpose(1, 2)
                macs0, _ = thop.profile(ph0, inputs=(h0, cond_s0), verbose=False)
                total_macs += float(macs0)
                s0_logits = ssm.head_s0(ssm.pred_s0(h0, cond_s0))
                c0 = s0_logits.argmax(dim=-1)
                curr_ll = ssm.tokenizer.inverse_step(curr_ll, c0, 0)
                ph1 = _PredHeadPair(ssm.pred_s1, ssm.head_s1).to(device).eval()
                h1 = curr_ll.flatten(2).transpose(1, 2)
                macs1, _ = thop.profile(ph1, inputs=(h1, cond_s1), verbose=False)
                total_macs += float(macs1)
                s1_logits = ssm.head_s1(ssm.pred_s1(h1, cond_s1))
                c1 = s1_logits.argmax(dim=-1)
                curr_ll = ssm.tokenizer.inverse_step(curr_ll, c1, 1)
                ph2 = _PredHeadPair(ssm.pred_s2, ssm.head_s2).to(device).eval()
                h2 = curr_ll.flatten(2).transpose(1, 2)
                macs2, _ = thop.profile(ph2, inputs=(h2, cond_s2), verbose=False)
                total_macs += float(macs2)
        except Exception:
            return 0.0
        return _macs_to_gflops(total_macs)


class SR3EvalWrapper(nn.Module):
    def __init__(self, sr3, ddim_steps=0, eta=0.0, lr_input_range: str = "01"):
        super().__init__()
        self.sr3, self.ddim_steps, self.eta = sr3, int(ddim_steps), float(eta)
        self.lr_input_range = str(lr_input_range).lower().strip()
        if self.lr_input_range not in ("01", "pm1"):
            raise ValueError("lr_input_range must be '01' or 'pm1'")

    def forward(self, lr):
        if self.lr_input_range == "01":
            lr_pm1 = lr * 2.0 - 1.0
        else:
            lr_pm1 = lr
        if self.ddim_steps > 0:
            out_pm1 = self.sr3(
                lr_pm1, sample_method="ddim", ddim_steps=self.ddim_steps, eta=self.eta
            )
        else:
            out_pm1 = self.sr3(lr_pm1)
        return (out_pm1 + 1.0) * 0.5

    def get_gflops(self, dummy_lr: torch.Tensor, device: torch.device) -> float:
        if thop is None:
            return 0.0
        self.eval()
        denoise_fn = self.sr3.diffusion.denoise_fn
        B = dummy_lr.size(0)
        d = dummy_lr.to(device)
        if self.lr_input_range == "01":
            lr_pm1 = d * 2.0 - 1.0
        else:
            lr_pm1 = d
        sz = int(getattr(self.sr3, "image_size", 48))
        out_hw = (sz, sz)
        lr_up = F.interpolate(lr_pm1, size=out_hw, mode="bilinear", align_corners=False)
        x_t = torch.randn(B, 1, out_hw[0], out_hw[1], device=device, dtype=torch.float32)
        inp = torch.cat([lr_up, x_t], dim=1)
        noise_level = torch.full((B, 1), 0.5, device=device, dtype=torch.float32)
        try:
            macs, _ = thop.profile(denoise_fn, inputs=(inp, noise_level), verbose=False)
        except Exception:
            return 0.0
        n = int(self.ddim_steps) if self.ddim_steps > 0 else int(self.sr3.diffusion.num_timesteps)
        return _macs_to_gflops(float(macs)) * n


class FastDiffusionEvalWrapper(nn.Module):
    def __init__(self, fd, num_steps=4, lr_input_range: str = "01"):
        super().__init__()
        self.fd = fd
        self.num_steps = int(num_steps)
        self.lr_input_range = str(lr_input_range).lower().strip()
        if self.lr_input_range not in ("01", "pm1"):
            raise ValueError("lr_input_range must be '01' or 'pm1'")

    def forward(self, lr):
        self.fd.num_inference_steps = self.num_steps
        x = lr * 2.0 - 1.0 if self.lr_input_range == "01" else lr
        out = self.fd(x)
        return (out + 1.0) * 0.5

    def get_gflops(self, dummy_lr: torch.Tensor, device: torch.device) -> float:
        if thop is None:
            return 0.0
        self.eval()
        fm = self.fd.diffusion
        denoise_fn = fm.denoise_fn
        B = dummy_lr.size(0)
        d = dummy_lr.to(device)
        x_lr = d * 2.0 - 1.0 if self.lr_input_range == "01" else d
        sz = int(getattr(self.fd, "image_size", 48))
        if x_lr.shape[-2:] != (sz, sz):
            x_lr = F.interpolate(x_lr, size=(sz, sz), mode="bilinear", align_corners=False)
        t = torch.full((B, 1), 0.5, device=device, dtype=x_lr.dtype)
        noise_level = fm._flow_t_to_sr3_noise_level(t)
        x_t = torch.randn_like(x_lr)
        inp = torch.cat([x_lr, x_t], dim=1)
        try:
            macs, _ = thop.profile(denoise_fn, inputs=(inp, noise_level), verbose=False)
        except Exception:
            return 0.0
        return _macs_to_gflops(float(macs)) * int(self.num_steps)


class VAREvalWrapper(nn.Module):
    def __init__(self, var, top_k=900, top_p=0.95, g_seed=None):
        super().__init__()
        self.var = var
        self.top_k, self.top_p, self.g_seed = top_k, top_p, g_seed

    def forward(self, lr):
        return self.var.autoregressive_infer_from_lr(
            lr, g_seed=self.g_seed, top_k=self.top_k, top_p=self.top_p
        )

    def get_gflops(self, dummy_lr: torch.Tensor, device: torch.device) -> float:
        if thop is None:
            return 0.0
        self.eval()
        var = self.var
        d = dummy_lr.to(device)
        B = d.size(0)
        try:
            macs_cond, _ = thop.profile(var.conditioner, inputs=(d,), verbose=False)
            with torch.no_grad():
                cf = var.conditioner(d)
                cond_BD = var.cond_pool(cf).flatten(1)
                lvl_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC
            macs_pool, _ = thop.profile(var.cond_pool, inputs=(cf,), verbose=False)
            stage = _VAROneStage(var).to(device).eval()
            total_stage_macs = 0.0
            for si, pn in enumerate(var.patch_nums):
                if si == 0:
                    next_token_map = (
                        cond_BD.unsqueeze(1).expand(B, var.first_l, -1)
                        + var.pos_start.expand(B, var.first_l, -1)
                        + lvl_pos[:, : var.first_l]
                    )
                else:
                    L = int(pn) ** 2
                    pos0 = _var_lvl_pos_start_for_stage(var, si)
                    h = torch.zeros(B, L, var.Cvae, device=device, dtype=cond_BD.dtype)
                    next_token_map = var.word_embed(h) + lvl_pos[:, pos0 : pos0 + L]
                macs_si, _ = thop.profile(stage, inputs=(next_token_map, cond_BD), verbose=False)
                total_stage_macs += float(macs_si)
            total_macs = float(macs_cond) + float(macs_pool) + total_stage_macs
        except Exception:
            return 0.0
        return _macs_to_gflops(total_macs)


class VARSREvalWrapper(nn.Module):
    def __init__(self, varsr, cfg=1.5, top_k=0, top_p=0.0, g_seed=None):
        super().__init__()
        self.varsr = varsr
        self.cfg = float(cfg)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.g_seed = g_seed

    def forward(self, lr):
        return self.varsr.generate(
            lr,
            cfg=self.cfg,
            top_k=self.top_k,
            top_p=self.top_p,
            g_seed=self.g_seed,
        )

    def get_gflops(self, dummy_lr: torch.Tensor, device: torch.device) -> float:
        if thop is None:
            return 0.0
        self.eval()
        v = self.varsr.var
        d = dummy_lr.to(device)
        B = d.size(0)
        try:
            lr_up = F.interpolate(d, size=(48, 48), mode="bilinear", align_corners=False)
            ce = _ConEmbeddingFeat(v.con_embedding).to(device).eval()
            macs_emb, _ = thop.profile(ce, inputs=(lr_up,), verbose=False)
            with torch.no_grad():
                lr_inp, _ = v.con_embedding(lr_up)
                lr_inp = lr_inp.view(B, v.C, -1).permute(0, 2, 1)
                sos = lr_inp.repeat(2, 1, 1)
                label_B = torch.zeros(B, dtype=torch.long, device=device)
                cond_BD = v.class_emb(
                    torch.cat(
                        (
                            label_B,
                            torch.full_like(label_B, fill_value=v.num_classes - 1),
                        ),
                        dim=0,
                    )
                )
                sos = torch.cat((sos, cond_BD.unsqueeze(1)), dim=1)
                lvl_pos = v.lvl_embed(v.lvl_1L)
            stage = _VARSROneStage(v).to(device).eval()
            freq_cur = 0
            total_stage_macs = 0.0
            for si, pn in enumerate(v.patch_nums):
                freqs_cis, freq_cur = _varsr_freqs_slice(v, si, int(pn), int(freq_cur))
                if si == 0:
                    next_token_map = (
                        sos.expand(2 * B, v.first_l, -1)
                        + v.pos_start.expand(2 * B, v.first_l, -1)
                        + lvl_pos[:, : v.first_l]
                    )
                else:
                    L = int(pn) ** 2
                    pos0 = _varsr_lvl_pos_start_for_stage(v, si)
                    h = torch.zeros(B, L, v.Cvae, device=device, dtype=cond_BD.dtype)
                    tm = v.word_embed(h.float()) + lvl_pos[:, pos0 : pos0 + L]
                    next_token_map = tm.repeat(2, 1, 1)
                freqs_cis = freqs_cis.to(device=device, dtype=next_token_map.dtype)
                macs_si, _ = thop.profile(
                    stage, inputs=(next_token_map, cond_BD, freqs_cis), verbose=False
                )
                total_stage_macs += float(macs_si)
            total_macs = float(macs_emb) + total_stage_macs
        except Exception:
            return 0.0
        return _macs_to_gflops(total_macs)


class VQGANTransformerEvalWrapper(nn.Module):
    def __init__(
        self,
        transformer,
        vqgan,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        greedy=False,
        start_token_id=0,
    ):
        super().__init__()
        self.transformer = transformer
        self.vqgan = vqgan
        self.vqgan.eval()
        self.vqgan.requires_grad_(False)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.greedy = greedy
        self.start_token_id = int(start_token_id)

    def forward(self, lr):
        tok = self.transformer.generate_tokens(
            lr,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            greedy=self.greedy,
            start_token_id=self.start_token_id,
        )
        return self.vqgan.decode_token_indices(tok)

    def get_gflops(self, dummy_lr: torch.Tensor, device: torch.device) -> float:
        if thop is None:
            return 0.0
        self.eval()
        tr = self.transformer
        T = int(tr.block_size)
        B = dummy_lr.size(0)
        d = dummy_lr.to(device)
        total_macs = 0.0
        try:
            for L in range(1, T):
                idx_L = torch.zeros(B, L, dtype=torch.long, device=device)
                macs_L, _ = thop.profile(tr, inputs=(d, idx_L), verbose=False)
                total_macs += float(macs_L)
        except Exception:
            return 0.0
        return _macs_to_gflops(total_macs)


class ResizeInterpEval(nn.Module):
    """Parameter-free LR→HR resize (32×16 → 48×48). Deterministic floor for Table I."""

    _ALIGN_MODES = frozenset({"linear", "bilinear", "bicubic", "trilinear"})

    def __init__(self, mode: str = "bilinear", out_hw: Tuple[int, int] = (48, 48)):
        super().__init__()
        self.mode = str(mode)
        self.out_hw = (int(out_hw[0]), int(out_hw[1]))

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        kw: Dict[str, object] = {"size": self.out_hw, "mode": self.mode}
        if self.mode in self._ALIGN_MODES:
            kw["align_corners"] = False
        return F.interpolate(lr, **kw)

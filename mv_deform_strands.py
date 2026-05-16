import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformMLP(nn.Module):
    def __init__(self, num_texels, C_dim=8, I_dim=8, max_disp=0.3, device='cuda'):
        """
        Deformation MLP indexed by texel ID (no explicit UV).
        - num_texels: H_up * W_up (e.g. 256*256 = 65536)
        - C_dim: per-frame latent dimension
        - I_dim: per-texel embedding dimension
        - max_disp: max 3D displacement (in canonical units)
        """
        super().__init__()
        self.max_disp = max_disp
        self.device = device

        # learnable embedding for texel indices [0..num_texels-1]
        self.texel_embed = nn.Embedding(num_texels, I_dim)
        nn.init.zeros_(self.texel_embed.weight)  # start with no spatial bias

        # MLP: [texel_embed, frame_code] -> 3D offset
        self.mlp = nn.Sequential(
            nn.Linear(I_dim + C_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

        # precompute tip weights w[j], j=0..198
        self.register_buffer('tip_weights', self._build_tip_weights(num_pts_rel=199, freeze_up_to=50 , device=device))

    @staticmethod
    def _build_tip_weights(num_pts_rel=199,freeze_up_to=10, device='cuda'):
        """
        freeze_up_to    # first 50 points: almost no motion

        Piecewise ramp along the strand:
        - near root: ~0
        - near tips: ~1
        """
        transition_len = 100 #50   # smooth ramp over next 50 points

        idx = torch.arange(num_pts_rel, device=device).float()  # [0..198]
        w = torch.zeros_like(idx)

        ramp_start = freeze_up_to
        ramp_end = min(num_pts_rel - 1, freeze_up_to + transition_len)

        ramp_mask = (idx >= ramp_start) & (idx <= ramp_end)
        w[ramp_mask] = (idx[ramp_mask] - ramp_start) / max(1.0, (ramp_end - ramp_start))
        w[idx > ramp_end] = 1.0
        w = w.clamp(0.0, 1.0)
        return w  # [199]

    def forward(self, pres, interested_idxes_up, frame_code):
        """
        Deform 'pres' in upsampled texel space for one frame.

        Inputs:
        - pres: [1, N_texels, 199, 3]  (from HAAR upsampling; offsets from roots)
        - interested_idxes_up: [N_up] indices of hair texels (non-bald)
        - frame_code: [C_dim] latent for this frame

        Returns:
        - upsampled_texture_deformed: [1, N_texels, 200, 3]
            (roots + deformed offsets, same layout as your original upsampled_texture)
        - delta_root_k_up: [N_up, 3]
            per-strand 3D offsets for valid texels (useful for regularization/monitoring)
        """

        # remove batch dim
        pres = pres[0]                              # [N_texels, 199, 3]
        # select only hair texels
        pres_valid = pres[interested_idxes_up]      # [N_up, 199, 3]

        # texel embeddings for valid indices
        texel_ids = interested_idxes_up.to(self.device).long()  # [N_up]
        texel_feat = self.texel_embed(texel_ids)                 # [N_up, I_dim]

        # frame code
        if frame_code.dim() == 1:
            frame_code = frame_code.unsqueeze(0)                 # [1, C_dim]
        N_up = texel_feat.shape[0]
        frame_exp = frame_code.expand(N_up, -1)                  # [N_up, C_dim]

        # input to MLP: [texel_embed, frame_code]
        inp = torch.cat([texel_feat, frame_exp], dim=-1)         # [N_up, I_dim + C_dim]

        # predict one 3D offset per valid texel, bounded with tanh
        raw = self.mlp(inp)                                      # [N_up, 3]
        offsets = self.max_disp * torch.tanh(raw)       # [N_up, 3]

        # tip weights w[j] in [0,1], shape [199]
        w = self.tip_weights                                     # [199]
        # broadcast to [N_up, 199, 3]
        delta_broadcast = offsets[:, None, :] * w[None, :, None]  # [N_up, 199, 3]

        # apply deformation to valid texels only (roots fixed; offsets deformed)
        pres_deformed_valid = pres_valid + delta_broadcast       # [N_up, 199, 3]

        # scatter back into full pres_deformed
        pres_deformed = pres.clone()                             # [N_texels, 199, 3]
        pres_deformed[interested_idxes_up] = pres_deformed_valid
        return pres_deformed, offsets

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np      

def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True):
    if num_encoding_functions == 0:
        return tensor

    encoding = [tensor] if include_input else []

    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0**0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        encoding.append(torch.sin(tensor * freq))
        encoding.append(torch.cos(tensor * freq))

    return encoding[0] if len(encoding) == 1 else torch.cat(encoding, dim=-1)



class DeformMLP(nn.Module):
    def __init__(
        self, 
        num_texels, 
        C_dim=8, 
        I_dim=8, 
        num_freqs=4, 
        hidden_dim=128, 
        max_disp_tangent=0.03,
        max_disp_normal=0.20,
        device='cuda'):
        """
        Deformation MLP indexed by texel ID (no explicit UV).
        - num_texels: H_up * W_up (e.g. 256*256 = 65536)
        - C_dim: per-frame latent dimension
        - I_dim: per-texel embedding dimension
        - num_freqs: positional encoding frequency bands

        """
        super().__init__()
        self.max_disp_tangent = max_disp_tangent
        self.max_disp_normal = max_disp_normal
        self.device = device
        self.num_freqs = num_freqs
        self.I_dim = I_dim
        self.offset_gain = nn.Parameter(torch.tensor(0.1))

        # learnable embedding for texel indices [0..num_texels-1]
        self.texel_embed = nn.Embedding(num_texels, I_dim)
        nn.init.zeros_(self.texel_embed.weight)  # start with no spatial bias


        self.frame_embed = nn.Sequential(
                    nn.Linear(C_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )

        # MeGA-style positional encoding: C_enc = 3 * (1 + 2*L_pos)
        C_pos = 3 #* (1 + 2 * num_freqs)      # include_input=True
        C_u_pos=1*  (1 + 2 * num_freqs)
        C_texel = I_dim
        C_frame = hidden_dim #C_dim

        # we'll feed [pos_enc, texel_embed, frame_code] to MLP
        in_dim = C_pos+ C_u_pos + C_texel + C_frame  # total MLP input dim

        self.deform_mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.Softplus(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Softplus(),
                    #nn.Linear(hidden_dim, 3),
                    nn.Linear(hidden_dim, 2),
                )
        # zero-init final layer so offsets start at 0
        #nn.init.zeros_(self.deform_mlp[-1].weight)
        #nn.init.zeros_(self.deform_mlp[-1].bias)
        nn.init.normal_(self.deform_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.deform_mlp[-1].bias, 0.0)

        # MLP: [texel_embed, frame_code] -> 3D offset
        #dims = [in_dim, 64, 64, 3]
        #self.deform_mlp = self._build_mlp(dims)
          
    @staticmethod
    def _compute_local_frames_with_root(pres_valid, eps=1e-8):
        """
        pres_valid: [N_strands, 199, 3]
            Root-relative strand points (without explicit root).

        Returns:
            T, N, B: [N_strands, 199, 3]
            Local frames aligned to each non-root point.
        """
        N_strands, S_pts, _ = pres_valid.shape

        # prepend explicit root at zero in local strand coordinates
        root_rel = torch.zeros(
            N_strands, 1, 3,
            device=pres_valid.device,
            dtype=pres_valid.dtype
        )
        strand_full = torch.cat([root_rel, pres_valid], dim=1)   # [N_strands, 200, 3]

        # forward differences on full strand
        tangents = strand_full[:, 1:] - strand_full[:, :-1]      # [N_strands, 199, 3]
        T = F.normalize(tangents, dim=-1, eps=eps)

        # choose stable reference axis
        ref_y = torch.tensor([0.0, 1.0, 0.0], device=pres_valid.device, dtype=pres_valid.dtype)
        ref_x = torch.tensor([1.0, 0.0, 0.0], device=pres_valid.device, dtype=pres_valid.dtype)

        ref = ref_y.view(1, 1, 3).expand_as(T).clone()
        parallel_mask = (T[..., 1].abs() > 0.9).unsqueeze(-1)
        ref = torch.where(parallel_mask, ref_x.view(1, 1, 3), ref)

        B = torch.cross(T, ref, dim=-1)
        B = F.normalize(B, dim=-1, eps=eps)

        N = torch.cross(B, T, dim=-1)
        N = F.normalize(N, dim=-1, eps=eps)

        return T, N, B


    def points_to_segments(self,pres_valid):
        """
        pres_valid: [N_strands, S_pts, 3]
            Root-relative point positions (without explicit root)

        Returns:
            segs: [N_strands, S_pts, 3]
            Segment vectors:
            segs[:,0]   = first point from root
            segs[:,j>0] = p_j - p_{j-1}
        """
        first = pres_valid[:, :1, :]
        rest = pres_valid[:, 1:, :] - pres_valid[:, :-1, :]
        segs = torch.cat([first, rest], dim=1)
        return segs


    def segments_to_points(self,segs):
        """
        segs: [N_strands, S_pts, 3]
            Segment vectors

        Returns:
            pres_valid: [N_strands, S_pts, 3]
            Root-relative point positions
        """
        return torch.cumsum(segs, dim=1)


    def compute_segment_frames(self,segs, eps=1e-8):
        """
        segs: [N_strands, S_pts, 3]
            Segment vectors

        Returns:
            T, N, B: [N_strands, S_pts, 3]
            Local frame per segment
        """
        T = F.normalize(segs, dim=-1, eps=eps)

        ref_y = torch.tensor([0.0, 1.0, 0.0], device=segs.device, dtype=segs.dtype)
        ref_x = torch.tensor([1.0, 0.0, 0.0], device=segs.device, dtype=segs.dtype)

        ref = ref_y.view(1, 1, 3).expand_as(T).clone()
        parallel_mask = (T[..., 1].abs() > 0.9).unsqueeze(-1)
        ref = torch.where(parallel_mask, ref_x.view(1, 1, 3), ref)

        B = torch.cross(T, ref, dim=-1)
        B = F.normalize(B, dim=-1, eps=eps)

        N = torch.cross(B, T, dim=-1)
        N = F.normalize(N, dim=-1, eps=eps)

        return T, N, B
    

    def forward(self, pres, interested_idxes_up, frame_code , step ,warmup_steps):
        """
            Inputs:
            - pres: [1, N_texels, S_pts, 3]  root-relative strand point positions
            - interested_idxes_up: [N_strands]
            - frame_code: [C_dim] or [1, C_dim]

            Returns:
            - pres_deformed: [N_texels, S_pts, 3]
            - offsets_xyz:   [N_strands, S_pts, 3]   segment-space offsets
        """
        pres = pres[0]  # [N_texels, S_pts, 3]
        interested_idxes_up = interested_idxes_up.to(self.device).long()

        # active strands only
        pres_valid = pres[interested_idxes_up]   # [N_strands, S_pts, 3]
        N_strands, S_pts, _ = pres_valid.shape

        # --------------------------------------------------
        # 1) convert root-relative points to segment vectors
        # --------------------------------------------------
        segs_valid = self.points_to_segments(pres_valid)   # [N_strands, S_pts, 3]

        # local frame from segment directions
        T, N, B = self.compute_segment_frames(segs_valid)

        # --------------------------------------------------
        # 2) build features
        # --------------------------------------------------
        seg_feat = segs_valid.reshape(-1, 3)   # use segment vectors as input

        u = torch.linspace(
            0.0, 1.0, S_pts,
            device=pres_valid.device,
            dtype=pres_valid.dtype
        )[None, :, None].expand(N_strands, S_pts, 1).reshape(-1, 1)

        u_enc = positional_encoding(
            u,
            num_encoding_functions=self.num_freqs,
            include_input=True,
            log_sampling=True,
        )

        texel_feat = self.texel_embed(interested_idxes_up)   # [N_strands, I_dim]
        texel_feat = texel_feat[:, None, :].expand(N_strands, S_pts, -1).reshape(-1, self.I_dim)

        if frame_code.dim() == 1:
            frame_code = frame_code.unsqueeze(0)
        frame_feat = self.frame_embed(frame_code).expand(N_strands * S_pts, -1)

        inp = torch.cat([seg_feat, u_enc, texel_feat, frame_feat], dim=-1)

        # --------------------------------------------------
        # 3) predict transverse segment offsets
        # --------------------------------------------------
        raw_coeff = self.deform_mlp(inp).view(N_strands, S_pts, 2)
        coeff = torch.clamp(raw_coeff, -1.0, 1.0)

        w = torch.linspace(
            0.0, 1.0, S_pts,
            device=pres_valid.device,
            dtype=pres_valid.dtype
        )
        w = 0.095 * (w ** 2)
        w = w[None, :, None]

        offsets_xyz =  (
            self.max_disp_normal * coeff[..., 0:1] * N +
            self.max_disp_normal * coeff[..., 1:2] * B
        ) * w

        # hard pin first few segments
        offsets_xyz[:, :10, :] = 0.0

        if step < warmup_steps:
            deform_alpha = 0.0
        else:
            ramp_len = 100
            deform_alpha = min((step - warmup_steps) / float(ramp_len), 1.0)
        offsets_xyz = deform_alpha * offsets_xyz

        # --------------------------------------------------
        # 4) deform segments, then reconstruct points
        # --------------------------------------------------
        segs_deformed_valid = segs_valid + offsets_xyz

        # optional: preserve segment lengths approximately
        seg_len = segs_valid.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        seg_dir = F.normalize(segs_deformed_valid, dim=-1, eps=1e-8)
        segs_deformed_valid = seg_dir * seg_len

        pres_deformed_valid = self.segments_to_points(segs_deformed_valid)

        # scatter back
        pres_deformed = pres.clone()
        pres_deformed[interested_idxes_up] = pres_deformed_valid

        return pres_deformed, offsets_xyz


    # def forward(self, pres, interested_idxes_up, frame_code):
    #     """
    #     Inputs:
    #     - pres: [1, N_texels, 199, 3]  (relative offsets from roots)
    #     - interested_idxes_up: [N_strands] indices of hair texels
    #     - frame_code: [C_dim] latent for this frame

    #     Returns:
    #     - pres_deformed: [N_texels, 199, 3]  (relative offsets after deformation)
    #     - offsets:       [N_strands, 199, 3]      (per-point offsets for valid texels)
    #     """
    #     # strip batch
    #     pres = pres[0]     
    #     # select only hair texels
    #     pres_valid = pres[interested_idxes_up]         # [N_strands, 199, 3]
    #     N_strands, S_pts, _ = pres_valid.shape
        
    #     # build strand-local frames from current canonical strand geometry
    #     # T: tangent, N: normal, B: binormal
    #     T, N, B = self._compute_local_frames_with_root(pres_valid)
        
    #     pts_rel = pres_valid.reshape(-1, 3)    
    #     # for the deformer to know where along the strand a point lies, beyond just its 3D position
    #     u = torch.linspace(
    #         0.0, 1.0, S_pts, device=pres_valid.device, dtype=pres_valid.dtype
    #     )[None, :, None].expand(N_strands, S_pts, 1).reshape(-1, 1)
    #     u_enc = positional_encoding(
    #         u,
    #         num_encoding_functions=self.num_freqs,
    #         include_input=True,
    #         log_sampling=True,
    #     )

    #     # # texel embedding: repeat for each point along strand
    #     texel_ids = interested_idxes_up.to(self.device).long()                                     # [N_strands]
    #     texel_feat = self.texel_embed(texel_ids)                                                   # [N_strands, I_dim]
    #     texel_feat = texel_feat[:, None, :].expand(N_strands, S_pts, -1).reshape(-1, self.I_dim)        # [N_strands*199, I_dim]

    #      # frame embedding: same for all points of this frame
    #     if frame_code.dim() == 1:
    #         frame_code = frame_code.unsqueeze(0) 
    #     frame_feat = self.frame_embed(frame_code)  # [1, hidden_dim]                                                   # [1, C_dim]
    #     frame_exp = frame_feat.expand(N_strands * S_pts, -1)                                                   # [N_strands, C_dim]

        
    #     # MLP input: [pos_enc, texel_feat, frame_exp]
    #     inp = torch.cat([pts_rel,u_enc, texel_feat, frame_exp], dim=-1)  # [N_strands*199, in_dim]

    #     raw_coeff = self.deform_mlp(inp)                                 # [N_strands*199, 3]
    #     coeff = raw_coeff.view(N_strands, S_pts, 2) #3)
    #     #coeff = torch.tanh(raw_coeff) 
    #     coeff = torch.clamp(coeff, -1.0, 1.0)                            # [N_strands,199,3]

    #     # bound offsets and apply root->tip weight
    #     #w = 0.8 *self.root_tip_weights[None, :, None]
    #     w = torch.linspace(0.0, 1.0, S_pts, device=pres_valid.device, dtype=pres_valid.dtype)# [1,199,1]
    #     w = 0.05 + 0.95 * (w ** 2)
    #     w=w[None, :, None]

    #     # convert local-frame coefficients into xyz offsets
    #     # offsets_xyz =  self.offset_gain *(
    #     #     self.max_disp_tangent * coeff[..., 0:1] * T +
    #     #     self.max_disp_normal  * coeff[..., 1:2] * N +
    #     #     self.max_disp_normal  * coeff[..., 2:3] * B
    #     # ) * w  # [N_strands, S_pts, 3]
    #     gain = F.softplus(self.offset_gain)  # positive
    #     coeff_std = coeff.std(dim=(0, 1))
    #     print("coeff_std:", coeff_std.detach().cpu(), "gain",gain.item())
        
    #     offsets_xyz = gain* (
    #         self.max_disp_normal * coeff[..., 0:1] * N +
    #         self.max_disp_normal * coeff[..., 1:2] * B
    #     )*w   # [N_strands, S_pts, 3]
         
    #     # final hard safety clamp
    #     # offset_norm = offsets_xyz.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    #     # max_norm = 0.03
    #     # scale = torch.clamp(max_norm / offset_norm, max=1.0)
    #     # offsets_xyz = offsets_xyz * scale
    #     offsets_xyz[:, :10, :] = 0.0
    #     pres_deformed_valid = pres_valid + offsets_xyz                 # [N_strands,199,3]

    #     pres_deformed = pres.clone()                               # [N_texels,199,3]
    #     pres_deformed[interested_idxes_up] = pres_deformed_valid

    #     return pres_deformed, offsets_xyz

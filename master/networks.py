import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat1, out_feat2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_feat, out_feat1, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(out_feat1),
            nn.ReLU(),
            nn.Conv3d(out_feat1, out_feat2, 3, padding=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(out_feat2),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )

    def forward(self, input):
        output = self.layers(input)
        return output


class ANCRNet(nn.Module):
    def __init__(self, n_feat, inshape, device, int_steps=7):
        super().__init__()
        self.device = device
        self.n_feat = n_feat
        self.inshape = inshape

        self.enc0 = ConvBlock(2, n_feat, n_feat)
        self.enc0_2 = ConvBlock(1, n_feat, n_feat)

        self.enc1 = ConvBlock(2 * n_feat, 3 * n_feat, 3 * n_feat)
        self.enc2 = ConvBlock(3 * n_feat, 4 * n_feat, 4 * n_feat)
        self.enc3 = ConvBlock(4 * n_feat, 8 * n_feat, 8 * n_feat)
        self.enc4 = ConvBlock(8 * n_feat, 16 * n_feat, 16 * n_feat)

        self.dec_def1 = ConvBlock(24 * n_feat, 12 * n_feat, 8 * n_feat)
        self.dec_def1_2 = nn.Sequential(
            nn.Conv3d(8 * n_feat, 4 * n_feat, 1),
            nn.ReLU(),
            nn.Conv3d(4 * n_feat, 3, 1)
        )

        self.dec_def2 = ConvBlock(12 * n_feat, 8 * n_feat, 6 * n_feat)
        self.dec_def2_2 = nn.Sequential(
            nn.Conv3d(6 * n_feat, 3 * n_feat, 1),
            nn.ReLU(),
            nn.Conv3d(3 * n_feat, 3, 1)
        )

        self.dec_def3 = nn.Sequential(
            ConvBlock(9 * n_feat, 6 * n_feat, 4 * n_feat),
            nn.Conv3d(4 * n_feat, 2 * n_feat, 1),
            nn.ReLU(),
            nn.Conv3d(2 * n_feat, 3, 1)
        )

        self.dec_seg1 = ConvBlock(24 * n_feat, 12 * n_feat, 8 * n_feat)

        self.dec_seg2 = ConvBlock(12 * n_feat, 8 * n_feat, 6 * n_feat)
        self.dec_seg2_2 = nn.Sequential(
            nn.Conv3d(6 * n_feat, 3 * n_feat, 1),
            nn.ReLU(),
            nn.Conv3d(3 * n_feat, 1, 1),
            nn.Sigmoid()
        )
        self.dec_app2_2 = nn.Sequential(
            nn.Conv3d(6 * n_feat, 3 * n_feat, 1),
            nn.Conv3d(3 * n_feat, 1, 1),
        )

        self.dec_seg3 = ConvBlock(9 * n_feat, 6 * n_feat, 4 * n_feat)
        self.dec_seg3_2 = nn.Sequential(
            nn.Conv3d(4 * n_feat, 2 * n_feat, 1),
            nn.ReLU(),
            nn.Conv3d(2 * n_feat, 1, 1),
            nn.Sigmoid()
        )
        self.dec_app3_2 = nn.Sequential(
            nn.Conv3d(4 * n_feat, 2 * n_feat, 1),
            nn.Conv3d(2 * n_feat, 1, 1),
        )

        self.dec_noCoMap = nn.Sequential(
            ConvBlock(6 * n_feat, 4 * n_feat, 2 * n_feat),
            nn.Conv3d(2 * n_feat, n_feat, 1),
            nn.ReLU(),
            nn.Conv3d(n_feat, 1, 1),
            nn.Sigmoid()
        )

        self.dec_appearanceMap = nn.Sequential(
            ConvBlock(6 * n_feat, 4 * n_feat, 2 * n_feat),
            nn.Conv3d(2 * n_feat, n_feat, 1),
            nn.Conv3d(n_feat, 1, 1)
        )

        self.maxPool = nn.MaxPool3d((1, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')

        down_shape1 = (inshape[0], inshape[1]//2, inshape[2]//2)
        down_shape2 = (inshape[0], inshape[1]//4, inshape[2]//4)
        self.integrate1 = VecInt(inshape, int_steps, device) if int_steps > 0 else None
        self.integrate2 = VecInt(down_shape1, int_steps, device) if int_steps > 0 else None
        self.integrate3 = VecInt(down_shape2, int_steps, device) if int_steps > 0 else None

        self.transformer1 = SpatialTransformer(inshape, device)
        self.transformer2 = SpatialTransformer(down_shape1, device)
        self.transformer3 = SpatialTransformer(down_shape2, device)

    def forward(self, moving, fixed, diff):
        inshape = self.inshape
        moving2 = F.interpolate(moving, size=(inshape[0], inshape[1]//2, inshape[2]//2), mode='trilinear')
        moving3 = F.interpolate(moving, size=(inshape[0], inshape[1]//4, inshape[2]//4), mode='trilinear')

        output0 = self.enc0(torch.cat((moving, fixed), dim=1))
        output0_maxPooled = self.maxPool(output0)

        output0_1 = self.enc0_2(diff)
        output0_1_maxPooled = self.maxPool(output0_1)

        output1 = self.enc1(torch.cat((output0_maxPooled, output0_1_maxPooled), dim=1))
        output1_maxPooled = self.maxPool(output1)

        output2 = self.enc2(output1_maxPooled)
        output2_maxPooled = self.maxPool(output2)

        output3 = self.enc3(output2_maxPooled)
        output3_maxPooled = self.maxPool(output3)

        output4 = self.enc4(output3_maxPooled)
        output4_upscaled = self.upsample(output4)

        # Branch 1
        output5 = self.dec_def1(torch.cat([output3, output4_upscaled], dim=1))
        output5_2 = self.dec_def1_2(output5)
        output5_upscaled = self.upsample(output5)

        output6 = self.dec_def2(torch.cat([output2, output5_upscaled], dim=1))
        output6_2 = self.dec_def2_2(output6)
        output6_upscaled = self.upsample(output6)

        output7 = self.dec_def3(torch.cat([output1, output6_upscaled], dim=1))

        # Implicite regularization as described in "Schwach ueberwachtes Lernen nichtlinearer medizinischer
        # Bildregistrierung mit neuronalen Faltungsnetzwerken" (S. Kuckertz, 2018): Deformation field u is calculated
        # on coarse grid and interpolated -> smoothing of deformation field
        v1 = F.interpolate(output7, scale_factor=(1, 2, 2), mode='trilinear')
        v2 = F.interpolate(output6_2, scale_factor=(1, 2, 2), mode='trilinear')
        v3 = F.interpolate(output5_2, scale_factor=(1, 2, 2), mode='trilinear')

        phi1 = self.integrate1(v1)
        phi2 = self.integrate2(v2)
        phi3 = self.integrate3(v3)

        # Branch 2
        segm5 = self.dec_seg1(torch.cat([output3, output4_upscaled], dim=1))
        segm5_upscaled = self.upsample(segm5)

        segm6 = self.dec_seg2(torch.cat([output2, segm5_upscaled], dim=1))
        segm6_upscaled = self.upsample(segm6)
        noCoMap3 = self.dec_seg2_2(segm6)
        appMap3 = self.dec_app2_2(segm6)

        segm7 = self.dec_seg3(torch.cat([output1, segm6_upscaled], dim=1))
        segm7_upscaled = self.upsample(segm7)
        noCoMap2 = self.dec_seg3_2(segm7)
        appMap2 = self.dec_app3_2(segm7)

        noCoMap1 = self.dec_noCoMap(torch.cat([output0, output0_1, segm7_upscaled], dim=1))
        appMap1 = self.dec_appearanceMap(torch.cat([output0, output0_1, segm7_upscaled], dim=1))

        out1 = self.transformer1(moving + noCoMap1 * appMap1, phi1)
        out2 = self.transformer2(moving2 + noCoMap2 * appMap2, phi2)
        out3 = self.transformer3(moving3 + noCoMap3 * appMap3, phi3)

        return v1, phi1, out1, v2, phi2, out2, v3, phi3, out3, \
               noCoMap1, noCoMap2, noCoMap3, appMap1, appMap2, appMap3


# Source: https://github.com/voxelmorph/voxelmorph.git
class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps, device):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape, device)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


# Source: https://github.com/voxelmorph/voxelmorph.git
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, device, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        # self.register_buffer('grid', grid)
        self.grid = grid.to(device)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
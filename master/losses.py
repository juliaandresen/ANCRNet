import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        intersections = (inputs * targets).sum(dim=[2, 3, 4])
        cardinalities = (inputs + targets).sum(dim=[2, 3, 4])
        dice = ((2 * intersections.sum(dim=1) + smooth) / (cardinalities.sum(dim=1) + smooth)).mean()
        return 1 - dice


class NCCLoss(nn.Module):
    def __init__(self):
        super(NCCLoss, self).__init__()

    def forward(self, moving, fixed):
        eps = 1e-3

        tmp1 = torch.square(torch.sum(fixed * moving, dim=[1, 2, 3, 4]))
        tmp2 = torch.sum(fixed**2, dim=[1, 2, 3, 4])
        tmp3 = torch.sum(moving**2, dim=[1, 2, 3, 4])
        ncc = (tmp1 + eps) / (tmp2 * tmp3 + eps)

        # Masked D_NCC
        d_ncc = 1 - ncc

        return d_ncc.mean()


class VectorFieldSmoothness(nn.Module):
    def __init__(self, h=1):
        super(VectorFieldSmoothness, self).__init__()
        self.h = h
        print('Warning: VectorFieldSmoothness is implemented only for isotropic voxel spacing!')

    def forward(self, deformation_field):

        h = self.h

        d1u1 = F.pad((deformation_field[:, 0:1, :, 2:, :] -
                      deformation_field[:, 0:1, :, :-2, :]), (0, 0, 1, 1, 0, 0), mode='replicate') / (2 * h)
        d2u1 = F.pad((deformation_field[:, 0:1, :, :, 2:] -
                      deformation_field[:, 0:1, :, :, :-2]), (1, 1, 0, 0, 0, 0), mode='replicate') / (2 * h)
        d3u1 = F.pad((deformation_field[:, 0:1, 2:, :, :] -
                      deformation_field[:, 0:1, :-2, :, :]), (0, 0, 0, 0, 1, 1), mode='replicate') / (2 * h)

        d1u2 = F.pad((deformation_field[:, 1:2, :, 2:, :] -
                      deformation_field[:, 1:2, :, :-2, :]), (0, 0, 1, 1, 0, 0), mode='replicate') / (2 * h)
        d2u2 = F.pad((deformation_field[:, 1:2, :, :, 2:] -
                      deformation_field[:, 1:2, :, :, :-2]), (1, 1, 0, 0, 0, 0), mode='replicate') / (2 * h)
        d3u2 = F.pad((deformation_field[:, 1:2, 2:, :, :] -
                      deformation_field[:, 1:2, :-2, :, :]), (0, 0, 0, 0, 1, 1), mode='replicate') / (2 * h)

        d1u3 = F.pad((deformation_field[:, 2:3, :, 2:, :] -
                      deformation_field[:, 2:3, :, :-2, :]), (0, 0, 1, 1, 0, 0), mode='replicate') / (2 * h)
        d2u3 = F.pad((deformation_field[:, 2:3, :, :, 2:] -
                      deformation_field[:, 2:3, :, :, :-2]), (1, 1, 0, 0, 0, 0), mode='replicate') / (2 * h)
        d3u3 = F.pad((deformation_field[:, 2:3, 2:, :, :] -
                      deformation_field[:, 2:3, :-2, :, :]), (0, 0, 0, 0, 1, 1), mode='replicate') / (2 * h)

        r = (d1u1.square() + d2u1.square() + d3u1.square() +
             d1u2.square() + d2u2.square() + d3u2.square() +
             d1u3.square() + d2u3.square() + d3u3.square()).sum(dim=[1, 2, 3, 4])

        return r.mean()
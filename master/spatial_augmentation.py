from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from master.displfields import warp_image_with_velo_field, compute_identity_grid

# Code written by Jan Ehrhardt (except ElasticRandomTransformPseudo3D)

def create_identity_grid_2d(size):
    assert len(size) == 2 or len(size) == 4
    batch_dim = 1 if len(size) == 2 else size[0]
    grid_size = (batch_dim, 1, size[-2], size[-1])
    id_theta = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])  # affine identity matrix
    id_theta = id_theta.expand(batch_dim, *id_theta.shape[1:])
    print(grid_size, id_theta.shape)
    identity_grid = F.affine_grid(id_theta, grid_size, align_corners=False)
    print(identity_grid.shape)
    return identity_grid


def generate_rand_deform_grid_2d(size=(10, 10), magnitude=(0.1, 0.1)):
    assert len(size) == 2 or len(size) == 4
    assert len(magnitude) == 2
    batch_dim = 1 if len(size) == 2 else size[0]
    grid_size = (batch_dim, size[-2], size[-1], 2)
    random_offsets = torch.randn(grid_size)
    random_magnitudes = torch.rand(grid_size)
    random_magnitudes[:,:,:,0] *= magnitude[0]
    random_magnitudes[:,:,:,1] *= magnitude[1]
    random_deform = random_offsets * random_magnitudes
    return random_deform


class Transform(ABC):
    """
    An abstract class of a ``Transform``.
    A transform is callable that processes ``data``.
    """

    @abstractmethod
    def __call__(self, data):
        """
        - ``data`` is a Numpy ndarray, PyTorch Tensor
        - the data shape can be:
        #. most of the pre-processing transforms expect: ``(num_channels, spatial_dim_1[, spatial_dim_2, ...])``,
             except that `AddChannel` expects (spatial_dim_1[, spatial_dim_2, ...]) and
             `AsChannelFirst` expects (spatial_dim_1[, spatial_dim_2, ...], num_channels)
        #. most of the post-processing transforms expect
             ``(batch_size, num_channels, spatial_dim_1[, spatial_dim_2, ...])``

        - the channel dimension is not omitted even if number of channels is one

        This method can optionally take additional arguments to help execute transformation operation.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class ElasticRandomTransform(Transform):
    """
    Implemented using :py:class:`torch.nn.functional.interpolate`.
    """

    def __init__(self, probability=0.5, grid_size=(10, 10), magnitude=(0.1, 0.1)) -> None:
        assert len(grid_size) == 2
        assert len(magnitude) == 2
        assert 0 <= probability <= 1
        self.probability = probability
        self.grid_size = grid_size
        self.magnitude = magnitude
        self.identity_grid = None
        self.deform_grid = None

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        if torch.rand(1).item() < self.probability:
            B, _, H, W = img.shape
            if self.identity_grid is None:
                self.identity_grid = create_identity_grid_2d(size=(H, W))

            identity_grid = self.identity_grid.expand(B, -1, -1, -1)
            self.deform_grid = generate_rand_deform_grid_2d(size=(B, self.grid_size[0],  self.grid_size[1], 2),
                                                            magnitude=self.magnitude)
            mapping_grid = F.interpolate((self.deform_grid).permute(0, 3, 1, 2), size=(H, W),
                                         mode='bicubic', align_corners=False).permute(0, 2, 3, 1)
            mapping_grid += identity_grid
            print(f'{mapping_grid.shape}, {img.shape}')
            out = F.grid_sample(img, mapping_grid, mode='bilinear', padding_mode='border', align_corners=False)
        else:
            out = img

        return out


class ElasticRandomTransform2D(Transform):
    """
    Implemented using :py:class:`torch.nn.functional.interpolate`.
    """

    def __init__(self, probability=0.5, grid_size=(10, 10), magnitude=(0.1, 0.1)) -> None:
        assert len(grid_size) == 2
        assert len(magnitude) == 2
        assert 0 <= probability <= 1
        self.probability = probability
        self.grid_size = grid_size
        self.magnitude = magnitude
        self.identity_grid = None
        self.last_velocity_field = None

    def _random_displacement(self, size, device=None, dtype=None):
        assert len(size) == 4
        B, C, H, W = size
        # generate coarse deformation grid
        grid_shape = (B, 2, self.grid_size[0], self.grid_size[1])
        random_magnitude_grid = torch.rand(grid_shape, device=device, dtype=dtype)
        random_magnitude_grid[:, 0, :, :] *= self.magnitude[0]
        random_magnitude_grid[:, 1, :, :] *= self.magnitude[1]
        random_deform_grid = torch.randn(grid_shape, device=device, dtype=dtype) * random_magnitude_grid
        # upsample to image resolution
        random_deform_grid = F.interpolate(random_deform_grid, size=(H, W), mode='bicubic', align_corners=False)
        return random_deform_grid

    def __call__(self, img, keep_transf=False):
        """
        Apply the transform to `img`.
        """
        if torch.rand(1).item() < self.probability:
            B, _, H, W = img.shape
            if self.identity_grid is None:
                self.identity_grid = compute_identity_grid(size=(H, W), device=img.device)

            identity_grid = self.identity_grid.expand(B, -1, -1, -1)
            if not keep_transf or self.last_velocity_field is None:
                self.last_velocity_field = self._random_displacement(size=img.shape, device=img.device)
            out_img = warp_image_with_velo_field(img, field=self.last_velocity_field, grid=identity_grid,
                                                 padding_mode='border')
        else:
            out_img = img

        return out_img


class ElasticRandomTransformPseudo3D(Transform):
    """
    Implemented using :py:class:`torch.nn.functional.interpolate`.
    """

    def __init__(self, probability=0.5, grid_size=(10, 10), magnitude=(0.1, 0.1)) -> None:
        assert len(grid_size) == 2
        assert len(magnitude) == 2
        assert 0 <= probability <= 1
        self.probability = probability
        self.grid_size = grid_size
        self.magnitude = magnitude
        self.identity_grid = None
        self.last_velocity_field = None
        self.last_label_velocity_field = None

    def _random_displacement(self, size, device=None, dtype=None):
        assert len(size) == 5
        B, C, _, H, W = size
        # generate coarse deformation grid
        grid_shape = (B, 2, self.grid_size[0], self.grid_size[1])
        random_magnitude_grid = torch.rand(grid_shape, device=device, dtype=dtype)
        random_magnitude_grid[:, 0, :, :] *= self.magnitude[0]
        random_magnitude_grid[:, 1, :, :] *= self.magnitude[1]
        random_deform_grid = torch.randn(grid_shape, device=device, dtype=dtype) * random_magnitude_grid
        # upsample to image resolution
        random_deform_grid = F.interpolate(random_deform_grid, size=(H, W), mode='bicubic', align_corners=False)
        random_deform_label_grid = F.interpolate(random_deform_grid, size=(H, W), mode='nearest')
        return random_deform_grid, random_deform_label_grid

    def __call__(self, img, label=None, keep_transf=False):
        """
        Apply the transform to `img`.
        """
        if torch.rand(1).item() < self.probability:
            B, _, D, H, W = img.shape
            if self.identity_grid is None:
                self.identity_grid = compute_identity_grid(size=(H, W), device=img.device)

            identity_grid = self.identity_grid.expand(B, -1, -1, -1)
            if not keep_transf or self.last_velocity_field is None:
                self.last_velocity_field, \
                self.last_label_velocity_field = self._random_displacement(size=img.shape, device=img.device)

            out_img = torch.zeros(img.shape)
            for slice in range(D):
                out_slice = warp_image_with_velo_field(img[:, :, slice, :, :], field=self.last_velocity_field, grid=identity_grid,
                                                       padding_mode='border')
                out_img[:, :, slice, :, :] = out_slice

            if label is not None:
                out_label = torch.zeros(label.shape)
                for slice in range(D):
                    out_slice = warp_image_with_velo_field(label[:, :, slice, :, :], field=self.last_label_velocity_field,
                                                           grid=identity_grid,
                                                           padding_mode='border')
                    out_label[:, :, slice, :, :] = out_slice
            else:
                out_label = None

        else:
            out_img = img
            if label is not None:
                out_label = label
            else:
                out_label = None

        return out_img, out_label

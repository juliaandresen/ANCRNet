import nibabel as nib
import numpy as np
import os
import torch
import torch.nn.functional as F
import warnings

from master.spatial_augmentation import ElasticRandomTransformPseudo3D
from scipy.ndimage import shift
from scipy.ndimage.interpolation import rotate
from skimage.exposure import equalize_adapthist
from skimage.morphology import erosion
from torch.utils.data import Dataset


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# Function to normalize input image to intensity range [0,1]
def normalize(image, path=None):
    if image.min() == 0 and image.max() == 0:
        return image
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                image = (image - image.min()) / (image.max() - image.min())
            except Warning as e:
                if path is not None:
                    print("Error in file {}...\n".format(path), e)
                else:
                    print("Error...", e)
        return image


# Dataset class to create dataset used during (main) trainnig of ANCR-Net
class MSLesionDataset(Dataset):
    def __init__(self, dataset, rootdir, train=False, slices_used=0.5, inshape=(5, 368, 512)):
        # dataset defines which images to use as test data, training fold 1 - 5 (indexed by 0 - 4)
        # rootdir: path to data

        if not dataset in [0, 1, 2, 3, 4]:
            raise Exception('Dataset must be one of the following: 0, 1, 2, 3, 4.')

        self.train = train
        self.images_baseline = {}
        self.images_followup = {}
        self.brain_masks = {}
        self.labels = {}
        self.patient_ids = {}
        self.inshape = inshape

        self.randomElasticDeformation = ElasticRandomTransformPseudo3D(probability=0.3, grid_size=(6, 10),
                                                                       magnitude=(0.01, 0.01))

        test_sets = [[19, 27, 37, 49, 68, 83, 91, 100],
                     [13, 20, 29, 39, 51, 69, 84, 94],
                     [15, 21, 30, 43, 52, 77, 88, 95],
                     [16, 24, 32, 47, 57, 74, 89, 96],
                     [18, 26, 35, 48, 61, 70, 90, 99]]

        pat_id = 0
        img_id = 0
        for dir, _, _ in os.walk(rootdir):
            if train:
                if dir != rootdir and int(dir[-3:]) not in test_sets[dataset]:
                    print(dir)

                    baseline_path = os.path.join(dir, 'flair_time01_on_middle_space.nii.gz')
                    followup_path = os.path.join(dir, 'flair_time02_on_middle_space.nii.gz')
                    brain_mask_path = os.path.join(dir, 'brain_mask.nii.gz')
                    segm_path = os.path.join(dir, 'ground_truth.nii.gz')

                    baseline_img = nib.load(baseline_path)
                    followup_img = nib.load(followup_path)
                    brain_mask = nib.load(brain_mask_path)
                    segm_img = nib.load(segm_path)

                    spacing = baseline_img.header.get_zooms()

                    baseline_vol = np.asarray(baseline_img.get_data())
                    followup_vol = np.asarray(followup_img.get_data())
                    brain_vol = np.asarray(brain_mask.get_data())
                    segm_vol = np.asarray(segm_img.get_data()).astype(np.float64)

                    size = baseline_vol.shape

                    baseline_vol = torch.tensor(baseline_vol).transpose(1, 2).transpose(0, 1)[None, ...]
                    followup_vol = torch.tensor(followup_vol).transpose(1, 2).transpose(0, 1)[None, ...]
                    brain_vol = torch.tensor(brain_vol)
                    segm_vol = torch.tensor(segm_vol).transpose(1, 2).transpose(0, 1)[None, ...]

                    baseline_vol, _ = new_resample(baseline_vol, spacing, size)
                    followup_vol, _ = new_resample(followup_vol, spacing, size)
                    segm_vol, _ = new_resample(segm_vol, spacing, size, mode='nearest')

                    tmp = np.where(brain_vol.sum(dim=[0, 1]))[0]
                    print('Using slices ' + str(tmp[0]) + ' to ' + str(tmp[-1]) + '.')

                    brain_vol = brain_vol.transpose(1, 2).transpose(0, 1)[None, ...]
                    brain_vol, _ = new_resample(1.*brain_vol, spacing, size, mode='nearest')

                    for slice in range(tmp[0] + inshape[0]//2, tmp[-1] - inshape[0]//2):

                        b = baseline_vol[0, slice - inshape[0]//2:slice + inshape[0]//2 + 1, ...].numpy()
                        f = followup_vol[0, slice - inshape[0]//2:slice + inshape[0]//2 + 1, ...].numpy()
                        s = segm_vol[0, slice - inshape[0]//2:slice + inshape[0]//2 + 1, ...].numpy()
                        m = brain_vol[0, slice - inshape[0]//2:slice + inshape[0]//2 + 1, ...].numpy()

                        # Use slices containing ground-truth segmentations twice with different orientations
                        if s.sum():
                            self.images_baseline[img_id] = b
                            self.images_followup[img_id] = f
                            self.labels[img_id] = s
                            self.brain_masks[img_id] = m
                            self.patient_ids[img_id] = pat_id
                            img_id += 1

                            self.images_baseline[img_id] = b[..., ::-1, :]
                            self.images_followup[img_id] = f[..., ::-1, :]
                            self.labels[img_id] = s[..., ::-1, :]
                            self.brain_masks[img_id] = m[..., ::-1, :]
                            self.patient_ids[img_id] = pat_id
                            img_id += 1
                        else:  # Use only given percentage of slices without new lesions
                            if np.random.choice([0, 1], 1, replace=True, p=[1 - slices_used, slices_used])[0] == 1:
                                self.images_baseline[img_id] = b
                                self.images_followup[img_id] = f
                                self.labels[img_id] = s
                                self.brain_masks[img_id] = m
                                self.patient_ids[img_id] = pat_id
                                img_id += 1
                pat_id += 1
            else:
                if dir != rootdir and int(dir[-3:]) in test_sets[dataset]:  # pat_id % 5 != dataset:
                    print(dir)

                    baseline_path = os.path.join(dir, 'flair_time01_on_middle_space.nii.gz')
                    followup_path = os.path.join(dir, 'flair_time02_on_middle_space.nii.gz')
                    brain_mask_path = os.path.join(dir, 'brain_mask.nii.gz')
                    segm_path = os.path.join(dir, 'ground_truth.nii.gz')

                    baseline_img = nib.load(baseline_path)
                    followup_img = nib.load(followup_path)
                    brain_mask = nib.load(brain_mask_path)
                    segm_img = nib.load(segm_path)

                    spacing = baseline_img.header.get_zooms()

                    baseline_vol = np.asarray(baseline_img.get_data())
                    followup_vol = np.asarray(followup_img.get_data())
                    brain_vol = np.asarray(brain_mask.get_data())
                    segm_vol = np.asarray(segm_img.get_data()).astype(np.float64)

                    size = baseline_vol.shape

                    baseline_vol = torch.tensor(baseline_vol).transpose(1, 2).transpose(0, 1)[None, ...]
                    followup_vol = torch.tensor(followup_vol).transpose(1, 2).transpose(0, 1)[None, ...]
                    brain_vol = torch.tensor(brain_vol)
                    segm_vol = torch.tensor(segm_vol).transpose(1, 2).transpose(0, 1)[None, ...]

                    baseline_vol, _ = new_resample(baseline_vol, spacing, size)
                    followup_vol, _ = new_resample(followup_vol, spacing, size)
                    segm_vol, _ = new_resample(segm_vol, spacing, size, mode='nearest')

                    brain_vol = brain_vol.transpose(1, 2).transpose(0, 1)[None, ...]
                    brain_vol, _ = new_resample(1. * brain_vol, spacing, size, mode='nearest')

                    for slice in range(inshape[0]//2, size[1] - inshape[0]//2):
                        b = baseline_vol[0, slice - inshape[0]//2:slice + inshape[0]//2 + 1, ...].numpy()
                        f = followup_vol[0, slice - inshape[0]//2:slice + inshape[0]//2 + 1, ...].numpy()
                        s = segm_vol[0, slice - inshape[0]//2:slice + inshape[0]//2 + 1, ...].numpy()
                        m = brain_vol[0, slice - inshape[0]//2:slice + inshape[0]//2 + 1, ...].numpy()

                        self.images_baseline[img_id] = b
                        self.images_followup[img_id] = f
                        self.labels[img_id] = s
                        self.brain_masks[img_id] = m
                        self.patient_ids[img_id] = pat_id
                        img_id += 1
                pat_id += 1

    def __len__(self):
        return len(self.images_baseline)

    def __getitem__(self, index):
        b = self.images_baseline[index]
        f = self.images_followup[index]
        m = self.brain_masks[index]
        s = self.labels[index]
        i = self.patient_ids[index]
        inshape = self.inshape

        if self.train:

            m_b = b.max()
            m_f = f.max()

            # Random noise
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.25, 0.25])[0] == 1:
                random_matrix = np.random.normal(0, 0.05*m_b, b.shape)
                b = b + random_matrix*m
                b[b < 0] = 0
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.25, 0.25])[0] == 1:
                random_matrix = np.random.normal(0, 0.05*m_f, f.shape)
                f = f + random_matrix*m
                f[f < 0] = 0

            # Random rotation
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.25, 0.25])[0] == 1:
                angle = np.round(np.random.uniform(-5, 5))
                b = rotate(b, angle, (1, 2), reshape=False)
                f = rotate(f, angle, (1, 2), reshape=False)
                m = rotate(m, angle, (1, 2), order=0, reshape=False)
                s = rotate(s, angle, (1, 2), order=0, reshape=False)

            # Random shift
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.125, 0.125])[0] == 1:
                shift_param = np.round(np.random.uniform(-5, 5))
                b = shift(b, (0, 0, shift_param))
                f = shift(f, (0, 0, shift_param))
                m = shift(m, (0, 0, shift_param), order=0)
                s = shift(s, (0, 0, shift_param), order=0)
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.125, 0.125])[0] == 1:
                shift_param = np.round(np.random.uniform(-5, 5))
                b = shift(b, (0, shift_param, 0))
                f = shift(f, (0, shift_param, 0))
                m = shift(m, (0, shift_param, 0), order=0)
                s = shift(s, (0, shift_param, 0), order=0)

            # Random brightness change
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.1, 0.1])[0] == 1:
                random_value = np.random.normal(0, 0.05 * m_b)
                b = b + m*random_value
                b[b < 0] = 0
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.1, 0.1])[0] == 1:
                random_value = np.random.normal(0, 0.05 * m_f)
                f = f + m*random_value
                f[f < 0] = 0

            # Random brightness gradient
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.1, 0.1])[0] == 1:
                random_value = np.random.normal(0, 0.05 * m_b)
                grad = np.reshape(np.tile(np.linspace(0, random_value, 512), 368), (1, 368, 512))
                b = b + m * grad
                b[b < 0] = 0
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.1, 0.1])[0] == 1:
                random_value = np.random.normal(0, 0.05 * m_f)
                grad = np.reshape(np.tile(np.linspace(0, random_value, 512), 368), (1, 368, 512))
                f = f + m * grad
                f[f < 0] = 0
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.1, 0.1])[0] == 1:
                random_value = np.random.normal(0, 0.05 * m_b)
                grad = np.reshape(np.reshape(np.tile(np.linspace(0, random_value, 368), 512), (512, 368)).T, (1, 368, 512))
                b = b + m * grad
                b[b < 0] = 0
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.1, 0.1])[0] == 1:
                random_value = np.random.normal(0, 0.05 * m_f)
                grad = np.reshape(np.reshape(np.tile(np.linspace(0, random_value, 368), 512), (512, 368)).T, (1, 368, 512))
                f = f + m * grad
                f[f < 0] = 0

            # Random adaptive equalization
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.125, 0.125])[0] == 1:
                b = normalize(b)
                f = normalize(f)
                for slice in range(inshape[0]):
                    b[slice, ...] = equalize_adapthist(b[slice, ...], clip_limit=0.02)
                    f[slice, ...] = equalize_adapthist(f[slice, ...], clip_limit=0.02)

            b = torch.from_numpy(b[None, None, ...].copy()).float()
            f = torch.from_numpy(f[None, None, ...].copy()).float()
            s = torch.from_numpy(s[None, ...].copy()).float()

            b, _ = self.randomElasticDeformation(b)
            f, _ = self.randomElasticDeformation(f)
            b = b[0, ...]
            f = f[0, ...]

        else:
            b = torch.from_numpy(b[None, ...].copy()).float()
            f = torch.from_numpy(f[None, ...].copy()).float()
            s = torch.from_numpy(s[None, ...].copy()).float()

        b = normalize(b)
        f = normalize(f)

        return {"image_baseline": b,
                "image_followup": f,
                "label": s,
                "patient_id": i}


class MSLesionDataset_Pretraining(Dataset):
    def __init__(self, dataset, rootdir, train=False, slices_used=0.5, inshape=(3, 368, 512)):
        # dataset defines which images to use as test data, training fold 1 - 5 (indexed by 0 - 4)

        if not dataset in [0, 1, 2, 3, 4]:
            raise Exception('Dataset must be one of the following: 0, 1, 2, 3, 4.')

        self.train = train
        self.images = {}
        self.brain_masks = {}
        self.lesion_placement_masks = {}
        self.labels = {}
        self.patient_ids = {}
        self.inshape = inshape

        self.randomElasticDeformation = ElasticRandomTransformPseudo3D(probability=1, grid_size=(6, 10),
                                                                       magnitude=(0.05, 0.05))

        test_sets = [[19, 27, 37, 49, 68, 83, 91, 100],
                     [13, 20, 29, 39, 51, 69, 84, 94],
                     [15, 21, 30, 43, 52, 77, 88, 95],
                     [16, 24, 32, 47, 57, 74, 89, 96],
                     [18, 26, 35, 48, 61, 70, 90, 99]]

        pat_id = 0
        img_id = 0
        for dir, _, _ in os.walk(rootdir):
            if train:
                if dir != rootdir and int(dir[-3:]) not in test_sets[dataset]:
                    print(dir)

                    # Use only baseline images for pretraining
                    baseline_path = os.path.join(dir, 'flair_time01_on_middle_space.nii.gz')
                    segm_path = os.path.join(dir, 'ground_truth.nii.gz')
                    brain_mask_path = os.path.join(dir, 'brain_mask.nii.gz')

                    baseline_img = nib.load(baseline_path)
                    segm_img = nib.load(segm_path)
                    brain_mask = nib.load(brain_mask_path)
                    spacing = baseline_img.header.get_zooms()

                    baseline_vol = np.asarray(baseline_img.get_data())
                    brain_vol = np.asarray(brain_mask.get_data())
                    segm_vol = np.asarray(segm_img.get_data()).astype(np.float64)

                    if segm_vol.sum():
                        continue

                    t = 0.15
                    lesion_placement_vol = brain_vol * (normalize(baseline_vol) > t)
                    lesion_placement_vol = erosion(erosion(lesion_placement_vol, np.ones((5, 5, 5))))

                    size = baseline_vol.shape

                    baseline_vol = torch.tensor(baseline_vol).transpose(1, 2).transpose(0, 1)[None, ...]
                    lesion_placement_vol = torch.tensor(lesion_placement_vol).transpose(1, 2).transpose(0, 1)[None, ...]
                    brain_vol = torch.tensor(brain_vol)

                    baseline_vol, _ = new_resample(baseline_vol, spacing, size)
                    lesion_placement_vol, _ = new_resample(lesion_placement_vol, spacing, size, mode='nearest')

                    tmp = np.where(brain_vol.sum(dim=[0, 1]))[0]
                    print('Using slices ' + str(tmp[0]) + ' to ' + str(tmp[-1]) + '.')

                    brain_vol = brain_vol.transpose(1, 2).transpose(0, 1)[None, ...]
                    brain_vol, _ = new_resample(1.*brain_vol, spacing, size, mode='nearest')

                    for slice in range(tmp[0] + inshape[0]//2, tmp[-1] - inshape[0]//2):

                        b = baseline_vol[0, slice - inshape[0]//2:slice + inshape[0]//2 + 1, ...].numpy()
                        m = brain_vol[0, slice - inshape[0]//2:slice + inshape[0]//2 + 1, ...].numpy()
                        p = lesion_placement_vol[0, slice - inshape[0]//2:slice + inshape[0]//2 + 1, ...].numpy()

                        if p.sum() and np.random.choice([0, 1], 1, replace=True, p=[1 - slices_used, slices_used])[0] == 1:
                            self.images[img_id] = b
                            self.labels[img_id] = np.zeros_like(b)
                            self.brain_masks[img_id] = m
                            self.lesion_placement_masks[img_id] = p
                            self.patient_ids[img_id] = pat_id
                            img_id += 1
                pat_id += 1
            else:
                if dir != rootdir and int(dir[-3:]) in test_sets[dataset]:
                    print(dir)

                    baseline_path = os.path.join(dir, 'flair_time01_on_middle_space.nii.gz')

                    if preprocessed:
                        brain_mask_path = os.path.join(dir, 'brain_mask.nii.gz')
                    else:
                        brain_mask_path = os.path.join(dir.replace('training_v2', 'training_v2_preprocessed'),
                                                       'brain_mask.nii.gz')

                    baseline_img = nib.load(baseline_path)
                    brain_mask = nib.load(brain_mask_path)

                    spacing = baseline_img.header.get_zooms()

                    baseline_vol = np.asarray(baseline_img.get_data())
                    brain_vol = np.asarray(brain_mask.get_data())

                    size = baseline_vol.shape

                    baseline_vol = torch.tensor(baseline_vol).transpose(1, 2).transpose(0, 1)[None, ...]
                    brain_vol = torch.tensor(brain_vol)

                    baseline_vol, _ = new_resample(baseline_vol, spacing, size)

                    brain_vol = brain_vol.transpose(1, 2).transpose(0, 1)[None, ...]
                    brain_vol, _ = new_resample(1. * brain_vol, spacing, size, mode='nearest')

                    for slice in range(inshape[0]//2, size[1] - inshape[0]//2):
                        b = baseline_vol[0, slice - inshape[0]//2:slice + inshape[0]//2 + 1, ...].numpy()
                        m = brain_vol[0, slice - inshape[0]//2:slice + inshape[0]//2 + 1, ...].numpy()

                        self.images[img_id] = b
                        self.brain_masks[img_id] = m
                        self.labels[img_id] = np.zeros_like(b)
                        self.patient_ids[img_id] = pat_id
                        img_id += 1
                pat_id += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        b = self.images[index]
        f = self.images[index].copy()
        m = self.brain_masks[index]
        s = self.labels[index]
        i = self.patient_ids[index]
        inshape = self.inshape

        if self.train:

            p = self.lesion_placement_masks[index]

            m_b = b.max()
            m_f = f.max()

            # Random noise
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.25, 0.25])[0] == 1:
                random_matrix = np.random.normal(0, 0.05*m_b, b.shape)
                b = b + random_matrix*m
                b[b < 0] = 0
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.25, 0.25])[0] == 1:
                random_matrix = np.random.normal(0, 0.05*m_f, f.shape)
                f = f + random_matrix*m
                f[f < 0] = 0

            # Random rotation
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.25, 0.25])[0] == 1:
                angle = np.round(np.random.uniform(-5, 5))
                b = rotate(b, angle, (1, 2), reshape=False)
                f = rotate(f, angle, (1, 2), reshape=False)
                m = rotate(m, angle, (1, 2), order=0, reshape=False)
                s = rotate(s, angle, (1, 2), order=0, reshape=False)
                p = rotate(p, angle, (1, 2), order=0, reshape=False)

            # Random shift
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.125, 0.125])[0] == 1:
                shift_param = np.round(np.random.uniform(-3, 3))
                b = shift(b, (0, 0, shift_param))
                f = shift(f, (0, 0, shift_param))
                m = shift(m, (0, 0, shift_param), order=0)
                s = shift(s, (0, 0, shift_param), order=0)
                p = shift(p, (0, 0, shift_param), order=0)
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.125, 0.125])[0] == 1:
                shift_param = np.round(np.random.uniform(-3, 3))
                b = shift(b, (0, shift_param, 0))
                f = shift(f, (0, shift_param, 0))
                m = shift(m, (0, shift_param, 0), order=0)
                s = shift(s, (0, shift_param, 0), order=0)
                p = shift(p, (0, shift_param, 0), order=0)

            # Random brightness change
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.1, 0.1])[0] == 1:
                random_value = np.random.normal(0, 0.05 * m_b)
                b = b + m*random_value
                b[b < 0] = 0
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.1, 0.1])[0] == 1:
                random_value = np.random.normal(0, 0.05 * m_f)
                f = f + m*random_value
                f[f < 0] = 0

            # Random brightness gradient
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.1, 0.1])[0] == 1:
                random_value = np.random.normal(0, 0.05 * m_b)
                grad = np.reshape(np.tile(np.linspace(0, random_value, 512), 368), (1, 368, 512))
                b = b + m * grad
                b[b < 0] = 0
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.1, 0.1])[0] == 1:
                random_value = np.random.normal(0, 0.05 * m_f)
                grad = np.reshape(np.tile(np.linspace(0, random_value, 512), 368), (1, 368, 512))
                f = f + m * grad
                f[f < 0] = 0
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.1, 0.1])[0] == 1:
                random_value = np.random.normal(0, 0.05 * m_b)
                grad = np.reshape(np.reshape(np.tile(np.linspace(0, random_value, 368), 512), (512, 368)).T, (1, 368, 512))
                b = b + m * grad
                b[b < 0] = 0
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.1, 0.1])[0] == 1:
                random_value = np.random.normal(0, 0.05 * m_f)
                grad = np.reshape(np.reshape(np.tile(np.linspace(0, random_value, 368), 512), (512, 368)).T, (1, 368, 512))
                f = f + m * grad
                f[f < 0] = 0

            # Random adaptive equalization
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.125, 0.125])[0] == 1:
                b = normalize(b)
                f = normalize(f)
                for slice in range(inshape[0]):
                    b[slice, ...] = equalize_adapthist(b[slice, ...], clip_limit=0.02)
                    f[slice, ...] = equalize_adapthist(f[slice, ...], clip_limit=0.02)

            b = normalize(b)
            f = normalize(f)

            # Random number of added new lesions
            n_lesions = np.random.randint(1, 6)
            # New lesions should only be added at positions defined by p
            pos1, pos2, pos3 = np.where(p)
            rand_idx = np.random.randint(0, len(pos1), n_lesions)

            # Add artificial new lesions
            for idx in rand_idx:
                c3 = pos1[idx]
                c2 = pos2[idx]
                c1 = pos3[idx]

                sigma1 = np.random.uniform(1, 7)
                sigma2 = np.random.uniform(1, 7)
                sigma3 = np.random.uniform(1, 7)

                new_lesion = np.array(gaussian(sigma1, c1, sigma2, c2, sigma3, c3, inshape))
                new_mask = new_lesion > 0.5

                f = f + np.random.uniform(0.3, 0.6) * new_lesion
                f[f > 1] = 1

                s = s + new_mask

            f = torch.from_numpy(f[None, None, ...].copy()).float()
            s = torch.from_numpy(s[None, None, ...].copy()).float()
            f, s = self.randomElasticDeformation(f, label=s)
            f = f[0, ...]
            s = s[0, ...]
            defField = self.randomElasticDeformation.last_velocity_field

        else:
            f = torch.from_numpy(f[None, ...].copy()).float()
            s = torch.from_numpy(s[None, ...].copy()).float()
            defField = None

        b = torch.from_numpy(b[None, ...].copy()).float()

        b = normalize(b)
        f = normalize(f)

        return {"image_baseline": b,
                "image_followup": f,
                "def_field": defField,
                "label": s,
                "patient_id": i}


# Function to resample MRIs to isotropic resolution and image size Dx368x512
def new_resample(image, spacing, size, mode='bilinear'):
    x, y, z = spacing
    W, H, D = size
    W_orig = W
    H_orig = H

    # Step 1: Resample such that lateral slices have 1:1 spacing
    if x < y:
        H = int(y / x * H)
        scale_x = 1
        scale_y = y / x
    elif x > y:
        W = int(x / y * W)
        scale_x = x / y
        scale_y = 1
    else:
        scale_x = 1
        scale_y = 1

    # Step 2: Rescale such that lateral slices have max. size of 368x512 pixels
    if W < 368 and H < 512:
        scale = min(368 / W, 512 / H)
    elif W > 368:
        scale = 368 / W
    elif H > 512:
        scale = 512 / H
    else:
        scale = 1

    if scale_x * scale * W_orig >= 369:
        print('Result different from original new resampling!')
        scale = 368 / W_orig / scale_x

    if scale_y * scale * H_orig >= 513:
        print('Result different from original new resampling!')
        scale = 512 / H_orig / scale_y

    image = F.interpolate(image, scale_factor=(scale_x * scale, scale_y * scale), mode=mode)

    _, _, W, H = image.shape

    # Step 3: Pad to 368x512xD
    p1, p2, p3, p4 = (0, 0, 0, 0)
    if W < 368:
        tmp = int((368 - W) / 2)
        p3, p4 = (tmp, tmp)
        if not W + p3 + p4 == 368:
            p4 += 1

    if H < 512:
        tmp = int((512 - H) / 2)
        p1, p2 = (tmp, tmp)
        if not H + p1 + p2 == 512:
            p2 += 1

    image = F.pad(image, (p1, p2, p3, p4), 'constant', 0)

    return image, {'step3p1': p1, 'step3p2': p2, 'step3p3': p3, 'step3p4': p4}


# Function to generate artificial new lesions (gaussian ellipsoids)
def gaussian(sigma1, c1, sigma2, c2, sigma3, c3, inshape):

    def gauss_fcn1(x):
        return -(x + inshape[1]/2 - c1 - inshape[2] // 2)**2 / float(2 * sigma1**2)

    def gauss_fcn2(x):
        return -(x + inshape[1]/2 - c2 - inshape[1] // 2)**2 / float(2 * sigma2**2)

    def gauss_fcn3(x):
        return -(x + inshape[0]/2 - c3 - inshape[0] // 2)**2 / float(2 * sigma3**2)

    gauss1 = torch.stack([torch.exp(torch.tensor(gauss_fcn1(x))) for x in range(inshape[2])])
    gauss2 = torch.stack([torch.exp(torch.tensor(gauss_fcn2(x))) for x in range(inshape[1])])
    gauss3 = torch.stack([torch.exp(torch.tensor(gauss_fcn3(x))) for x in range(inshape[0])])

    gauss1 = gauss1.repeat(inshape[0], inshape[1], 1)
    gauss2 = gauss2.unsqueeze(0).transpose(0, 1).repeat(inshape[0], 1, inshape[2])
    gauss3 = gauss3.unsqueeze(0).unsqueeze(0).transpose(0,2).repeat(1, inshape[1], inshape[2])

    gauss = gauss1*gauss2*gauss3
    return (gauss - gauss.min()) / (gauss.max() - gauss.min())
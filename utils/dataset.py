import os
import random
import numpy as np
import scipy.io as io
import sklearn.model_selection
from sklearn import preprocessing
import tifffile as tiff
import torch
import torch.utils.data
import torch.nn.functional as F


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        return io.loadmat(dataset)
    elif ext in ('.tif', '.tiff'):
        return tiff.imread(dataset)
    else:
        raise ValueError("Unknown file format: {}".format(ext))


def load_mat_hsi(dataset_name, dataset_dir, norm='normband'):
    """ load HSI.mat dataset """
    available_sets = [
        'sa', 'pu', 'whulk', 'hrl', 'Loukia', 'Dioni',
        'Houston18', 'Houston13',
        'MFF_SD', 'MFF_TD1', 'MFF_TD2', 'MJG_SD', 'MJG_TD1', 'MJG_TD2'
    ]

    img = None
    gt = None
    labels = None

    if dataset_name == 'sa':
        data = io.loadmat(os.path.join(dataset_dir, dataset_name, "Salinas_corrected.mat"))
        img = data['salinas_corrected']
        gt_data = io.loadmat(os.path.join(dataset_dir, dataset_name, "Salinas_gt.mat"))
        gt = gt_data['salinas_gt']
        labels = [
            "Undefined", "Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow",
            "Fallow_rough_plow", "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained",
            "Soil_vinyard_develop", "Corn_senesced_green_weeds", "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk", "Lettuce_romaine_6wk", "Lettuce_romaine_7wk",
            "Vinyard_untrained", "Vinyard_vertical_trellis"
        ]

    elif dataset_name == 'pu':
        data = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU.mat"))
        img = data['paviaU']
        gt_data = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU_gt.mat"))
        gt = gt_data['paviaU_gt']
        labels = [
            "Undefined", "Asphalt", "Meadows", "Gravel", "Trees",
            "Painted metal sheets", "Bare Soil", "Bitumen",
            "Self-Blocking Bricks", "Shadows"
        ]

    elif dataset_name == 'whulk':
        data = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou.mat"))
        img = data['WHU_Hi_LongKou']
        gt_data = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou_gt.mat"))
        gt = gt_data['WHU_Hi_LongKou_gt']
        labels = [
            'Undefined', 'Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean',
            'Narrow-leaf soybean', 'Rice', 'Water', 'Roads and houses', 'Mixed weed'
        ]

    elif dataset_name == 'Loukia':
        img = open_file(os.path.join(dataset_dir, 'Loukia.mat'))['ori_data']
        gt = open_file(os.path.join(dataset_dir, 'Loukia_gt_out68.mat'))['map']
        labels = [
            'Undefined', 'Dense Urban Fabric', 'Mineral Extraction Sites',
            'Non Irrigated Arable Land', 'Fruit Trees', 'Olive Groves',
            'Coniferous Forest', 'Dense Sclerophyllous Vegetation',
            'Sparce Sclerophyllous Vegetation', 'Sparcely Vegetated Areas',
            'Rocks and Sand', 'Water', 'Coastal Water'
        ]

    elif dataset_name == 'Dioni':
        img = open_file(os.path.join(dataset_dir, 'Dioni.mat'))['ori_data']
        gt = open_file(os.path.join(dataset_dir, 'Dioni_gt_out68.mat'))['map']
        labels = [
            'Undefined', 'Dense Urban Fabric', 'Mineral Extraction Sites',
            'Non Irrigated Arable Land', 'Fruit Trees', 'Olive Groves',
            'Coniferous Forest', 'Dense Sclerophyllous Vegetation',
            'Sparce Sclerophyllous Vegetation', 'Sparcely Vegetated Areas',
            'Rocks and Sand', 'Water', 'Coastal Water'
        ]

    elif dataset_name == 'Houston18':
        img = open_file(os.path.join(dataset_dir, 'Houston18.mat'))['ori_data']
        gt = open_file(os.path.join(dataset_dir, 'Houston18_7gt.mat'))['map']
        labels = ['Undefined', "Grass healthy", "Grass stressed", "Trees", "Water",
                  "Residential buildings", "Non-residential buildings", "Road"]

    elif dataset_name == 'Houston13':
        img = open_file(os.path.join(dataset_dir, 'Houston13.mat'))['ori_data']
        gt = open_file(os.path.join(dataset_dir, 'Houston13_7gt.mat'))['map']
        labels = ['Undefined', "Grass healthy", "Grass stressed", "Trees", "Water",
                  "Residential buildings", "Non-residential buildings", "Road"]

    elif dataset_name in ('MFF_SD', 'MJG_SD'):
        mat = open_file(os.path.join(dataset_dir, 'MFF_SD.mat'))
        img = mat.get('ori_data', mat.get('img', mat.get('data')))
        gt = mat.get('map', mat.get('gt'))
        labels = ['Undefined', 'Larch', 'Mongolian pine', 'Korean pine', 'Spruce', 'Broad-leaved trees']

    elif dataset_name in ('MFF_TD1', 'MJG_TD1'):
        mat = open_file(os.path.join(dataset_dir, 'MFF_TD1.mat'))
        img = mat.get('ori_data', mat.get('img', mat.get('data')))
        gt = mat.get('map', mat.get('gt'))
        labels = ['Undefined', 'Larch', 'Mongolian pine', 'Korean pine', 'Spruce', 'Broad-leaved trees']

    elif dataset_name in ('MFF_TD2', 'MJG_TD2'):
        mat = open_file(os.path.join(dataset_dir, 'MFF_TD2.mat'))
        img = mat.get('ori_data', mat.get('img', mat.get('data')))
        gt = mat.get('map', mat.get('gt'))
        labels = ['Undefined', 'Larch', 'Mongolian pine', 'Korean pine', 'Spruce', 'Broad-leaved trees']
    else:
        # Fallback to direct filename match in dataset_dir
        mat_path = os.path.join(dataset_dir, f"{dataset_name}.mat")
        if os.path.exists(mat_path):
            mat = open_file(mat_path)
            img = mat.get('ori_data', mat.get('img', mat.get('data')))
            gt = mat.get('map', mat.get('gt'))
            labels = ['Undefined'] + [str(i) for i in range(1, int(np.nanmax(gt)) + 1)]
        else:
            raise ValueError(f"Unknown dataset {dataset_name} in {dataset_dir}")

    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print("warning: nan values found in dataset {}, using 0 to replace them".format(dataset_name))
        img[nan_mask] = 0
        gt[nan_mask] = 0

    if norm == 'normband':
        img = np.asarray(img, dtype='float32')
        m, n, d = img.shape[0], img.shape[1], img.shape[2]
        img_ori = img.reshape((m * n, -1))
        index = np.where(img_ori.sum(axis=-1) != 0)
        img = img_ori[index]
        max_val = img.max() if img.max() > 0 else 1.0
        img = img / max_val
        img_temp = np.sqrt(np.asarray((img ** 2).sum(1)))
        img_temp = np.expand_dims(img_temp, axis=1)
        img_temp = img_temp.repeat(d, axis=1)
        img_temp[img_temp == 0] = 1
        img = img / img_temp
        img_ori[index] = img
        img = np.reshape(img_ori, (m, n, -1))
    elif norm == 'minmax':
        img = np.asarray(img, dtype=np.float32)
        min_v = np.min(img)
        max_v = np.max(img)
        denom = (max_v - min_v) if max_v > min_v else 1.0
        img = (img - min_v) / denom
        mean_by_c = np.mean(img, axis=(0, 1))
        for c in range(img.shape[-1]):
            img[:, :, c] = img[:, :, c] - mean_by_c[c]
    elif norm == 'std':
        meanhsi = np.mean(np.reshape(img, -1))
        sigmahsi = np.sqrt(np.var(np.reshape(img, -1)))
        img = (img - meanhsi) / (sigmahsi + 1e-8)
    elif norm == 'ln':
        img = img / (np.sqrt(np.sum(img ** 2, axis=2, keepdims=True)) + 1e-9)
    elif norm == 'sklearn':
        data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:]))
        data_scaler = preprocessing.scale(data)
        img = data_scaler.reshape(img.shape[0], img.shape[1], img.shape[2])

    gt = gt.astype('int') - 1
    labels = labels[1:]

    return img, gt, labels


def sample_gt(gt, train_size, seed, mode='random'):
    indices = np.where(gt >= 0)
    X = list(zip(*indices))
    y = gt[indices].ravel()
    train_gt = np.full_like(gt, fill_value=-1)
    test_gt = np.full_like(gt, fill_value=-1)
    if train_size > 1:
        train_size = int(train_size)

    if mode == 'random':
        if train_size == 1:
            random.seed(seed)
            random.shuffle(X)
            train_indices = [list(t) for t in zip(*X)]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt = []
        else:
            train_indices, test_indices = sklearn.model_selection.train_test_split(
                X, train_size=train_size, stratify=y, random_state=seed
            )
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
    else:
        raise ValueError(f"Sampling mode '{mode}' not implemented.")

    return train_gt, test_gt


def apply_ars_noise(tensor, noise_type='randomrc_gauss', sigma=0.05):
    """
    Apply Adaptability Reinforcement Strategy (ARS) noise perturbations.
    Supported types:
      - 'none': no noise added
      - 'gauss': Gaussian noise N(0, sigma^2) with sigma=0.05
      - 'randomrc': Random rotation & cropping/scaling (RandomRC)
      - 'radiation': Radiation noise (illumination scale alpha in [0.9, 1.1] + beta * N(0, 1))
      - 'randomrc_gauss': Both RandomRC and Gaussian noise (optimal in paper for MFF & Loukia)
      - 'all' / 'randomrc_gauss_radiation': RandomRC + Gauss + Radiation (optimal for Houston 2018)
    Accepts 4D (B, C, H, W) or 5D (B, 1, C, H, W) PyTorch tensors.
    """
    if noise_type is None or noise_type == 'none':
        return tensor

    out = tensor.clone()

    # 1. RandomRC (random rotation & spatial reflection/cropping)
    if 'randomrc' in noise_type or noise_type == 'all':
        # Apply random spatial flips/rotations per batch item
        k = random.randint(0, 3)
        if k > 0:
            out = torch.rot90(out, k, dims=[-2, -1])
        if random.random() > 0.5:
            out = torch.flip(out, dims=[-1])
        if random.random() > 0.5:
            out = torch.flip(out, dims=[-2])

    # 2. Radiation noise: alpha * x + beta * noise
    if 'radiation' in noise_type or noise_type == 'all':
        alpha = random.uniform(0.9, 1.1)
        beta = 0.05
        rad_noise = torch.randn_like(out) * beta
        out = alpha * out + rad_noise

    # 3. Gaussian noise: N(0, sigma^2) with sigma=0.05
    if 'gauss' in noise_type or noise_type == 'all':
        gauss_noise = torch.randn_like(out) * sigma
        out = out + gauss_noise

    return out


class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, image, gt, patch_size, data_aug=True,
                 flip_augmentation=False, radiation_augmentation=False,
                 mixture_augmentation=False, noise_type=None):
        super().__init__()
        self.data = image
        self.label = gt
        self.data_aug = data_aug
        self.patch_size = patch_size
        self.ps = self.patch_size // 2
        self.flip_augmentation = flip_augmentation
        self.radiation_augmentation = radiation_augmentation
        self.mixture_augmentation = mixture_augmentation
        self.noise_type = noise_type

        mask = np.ones_like(self.label)
        mask[self.label < 0] = 0
        x_pos, y_pos = np.nonzero(mask)

        self.indices = np.array([
            (x, y) for x, y in zip(x_pos, y_pos)
            if (self.ps < x < self.data.shape[0] - self.ps) and
               (self.ps < y < self.data.shape[1] - self.ps)
        ])
        self.labels = [self.label[x, y] for x, y in self.indices]

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=0.05):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    @staticmethod
    def gaussian_noise(data, sigma=0.05):
        noise = np.random.normal(loc=0., scale=sigma, size=data.shape)
        return data + noise

    def hsi_augment(self, data):
        do_augment = np.random.random()
        if do_augment > 0.5:
            prob = np.random.random()
            if 0 <= prob <= 0.2:
                data = np.fliplr(data)
            elif 0.2 < prob <= 0.4:
                data = np.flipud(data)
            elif 0.4 < prob <= 0.6:
                data = np.rot90(data, k=1)
            elif 0.6 < prob <= 0.8:
                data = np.rot90(data, k=2)
            elif 0.8 < prob <= 1.0:
                data = np.rot90(data, k=3)
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x, y]

        if self.data_aug:
            data = self.hsi_augment(data)

        if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
            data, _ = self.flip(data, self.label[x1:x2, y1:y2])
        if self.radiation_augmentation and np.random.random() < 0.5:
            data = self.radiation_noise(data)

        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        data = data.unsqueeze(0)  # (1, C, H, W)

        return data, label

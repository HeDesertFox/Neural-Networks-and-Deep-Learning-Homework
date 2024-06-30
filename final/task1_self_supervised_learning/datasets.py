from torchvision import datasets
import torchvision.transforms as transforms


class DataSet:
    """Dataset used for SimCLR experiment"""

    def __init__(self, data_folder) -> None:
        self.data_folder = data_folder

    def data_aug(self, size=32, kernel_size=(3, 3), s=1, sigma=(0.1, 2.0), n_views=2):
        """Get the data augmentation transform for the dataset,
        For CIFAR10, size shall be set to 32,
        For stl10, size shall be set to 96.
        """
        self._size = size

        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        _base_data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomResizedCrop(size=size, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
            ]
        )
        self.data_transform = lambda e: [
            _base_data_transform(e) for _ in range(n_views)
        ]

    def get_dataset(self, dataset_type: str):
        if not dataset_type in ["CIFAR10", "stl10"]:
            raise ValueError

        if dataset_type == "CIFAR10":
            if self._size != 32:
                raise ValueError("The size for CIFAR10 shall be 32")
            self.dataset = datasets.CIFAR10(
                root=self.data_folder,
                transform=self.data_transform,
                download=True,
                train=True,
            )
        elif dataset_type == "stl10":
            if self._size != 96:
                raise ValueError("The size for stl10 shall be 96")
            self.dataset = datasets.STL10(
                root=self.data_folder,
                transform=self.data_transform,
                download=True,
                split="unlabeled",
            )

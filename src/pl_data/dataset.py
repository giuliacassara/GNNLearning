
from omegaconf import ValueNode
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST


class MyDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode, train: bool, **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.train = train

        self.mnist = FashionMNIST(path, train=train, download=True, **kwargs)

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, index: int):
        return self.mnist[index]

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=})"
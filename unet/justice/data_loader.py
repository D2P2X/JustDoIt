import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith("label")]
        lst_input = [f for f in lst_data if f.startswith("input")]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label / 255.0
        input = input / 255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {"input": input, "label": label}

        if self.transform:
            data = self.transform(data)

        self.data = data
        return data

    def show(self):
        fig = plt.figure()
        ax = plt.subplot(1, 4, 1)
        ax.set_title("label")
        plt.imshow(self.data["label"])
        ax = plt.subplot(1, 4, 2)
        ax.set_title("input")
        plt.imshow(self.data["input"])


class ToTensor(object):
    def __call__(self, data):
        label, input = data["label"], data["input"]
        # print(label.shape, input.shape)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)
        # print(label.shape, input.shape)

        data = {"label": torch.from_numpy(label), "input": torch.from_numpy(input)}
        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data["label"], data["input"]

        input = (input - self.mean) / self.std

        data = {"label": label, "input": input}
        return data


class RandomFlip(object):
    def __call__(self, data):
        label, input = data["label"], data["input"]
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {"label": label, "input": input}
        return data


if __name__ == "__main__":

    dir_save_train = "datasets/train"
    # d = Dataset(data_dir="datasets/train")
    # print(d.__len__)
    # d.__getitem__(index=0)
    # d.show()
    # plt.show()
    transform = transforms.Compose(
        [Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()]
    )
    dataset_train = Dataset(data_dir=dir_save_train, transform=transform)
    data = dataset_train.__getitem__(0)  # 한 이미지 불러오기
    input = data["input"]
    label = data["label"]

    # 불러온 이미지 시각화
    plt.subplot(122)
    plt.hist(label.flatten(), bins=20)
    plt.title("label")

    plt.subplot(121)
    plt.hist(input.flatten(), bins=20)
    plt.title("input")

    plt.tight_layout()
    plt.show()

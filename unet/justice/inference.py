import os
import torch
from data_loader import Normalization, RandomFlip, ToTensor, Dataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from model import UNet
import torch.nn as nn
import numpy as np
from train import load
import matplotlib.pyplot as plt

if __name__ == "__main__":
    lr = 1e-3
    batch_size = 4
    num_epoch = 20

    base_dir = "./save_unet"
    data_dir = "./datasets"
    ckpt_dir = os.path.join(base_dir, "checkpoint")
    log_dir = os.path.join(base_dir, "log")

    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, "test"), transform=transform)
    loader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=8
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    net = UNet().to(device)

    fn_loss = nn.BCEWithLogitsLoss().to(device)

    optim = torch.optim.Adam(net.parameters(), lr=lr)

    fn_tonumpy = lambda x: x.to("cpu").detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

    result_dir = os.path.join(base_dir, "result")
    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir, "png"))
        os.makedirs(os.path.join(result_dir, "numpy"))

    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            label = data["label"].to(device)
            input = data["input"].to(device)

            output = net(input)

            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print(
                "TEST: BATCH %04d / %04d | LOSS %.4f"
                % (batch, num_batch_test, np.mean(loss_arr))
            )

            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                plt.imsave(
                    os.path.join(result_dir, "png", "label_%04d.png" % id),
                    label[j].squeeze(),
                    cmap="gray",
                )
                plt.imsave(
                    os.path.join(result_dir, "png", "input_%04d.png" % id),
                    input[j].squeeze(),
                    cmap="gray",
                )
                plt.imsave(
                    os.path.join(result_dir, "png", "output_%04d.png" % id),
                    output[j].squeeze(),
                    cmap="gray",
                )

                np.save(
                    os.path.join(result_dir, "numpy", "label_%04d.npy" % id),
                    label[j].squeeze(),
                )
                np.save(
                    os.path.join(result_dir, "numpy", "input_%04d.npy" % id),
                    input[j].squeeze(),
                )
                np.save(
                    os.path.join(result_dir, "numpy", "output_%04d.npy" % id),
                    output[j].squeeze(),
                )

    print(
        "AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f"
        % (batch, num_batch_test, np.mean(loss_arr))
    )

    lst_data = os.listdir(os.path.join(result_dir, "numpy"))

    lst_label = [f for f in lst_data if f.startswith("label")]
    lst_input = [f for f in lst_data if f.startswith("input")]
    lst_output = [f for f in lst_data if f.startswith("output")]

    lst_label.sort()
    lst_input.sort()
    lst_output.sort()

    id = 0

    label = np.load(os.path.join(result_dir, "numpy", lst_label[id]))
    input = np.load(os.path.join(result_dir, "numpy", lst_input[id]))
    output = np.load(os.path.join(result_dir, "numpy", lst_output[id]))

    plt.figure(figsize=(8, 6))
    plt.subplot(131)
    plt.imshow(input, cmap="gray")
    plt.title("input")

    plt.subplot(132)
    plt.imshow(label, cmap="gray")
    plt.title("label")

    plt.subplot(133)
    plt.imshow(output, cmap="gray")
    plt.title("output")

    plt.show()

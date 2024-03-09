import os
import torch
from data_loader import Normalization, RandomFlip, ToTensor, Dataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from model import UNet
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save(
        {"net": net.state_dict(), "optim": optim.state_dict()},
        "%s/model_epoch%d.pth" % (ckpt_dir, epoch),
    )


def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    dict_model = torch.load("%s/%s" % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model["net"])
    optim.load_state_dict(dict_model["optim"])
    epoch = int(ckpt_lst[-1].split("epoch")[1].split(".pth")[0])

    return net, optim, epoch


if __name__ == "__main__":

    lr = 1e-3
    batch_size = 4
    num_epoch = 20

    base_dir = "./save_unet"
    data_dir = "./datasets"
    ckpt_dir = os.path.join(base_dir, "checkpoint")
    log_dir = os.path.join(base_dir, "log")

    transform = transforms.Compose(
        [Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()]
    )

    dataset_train = Dataset(
        data_dir=os.path.join(data_dir, "train"), transform=transform
    )
    loader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=8
    )

    dataset_val = Dataset(data_dir=os.path.join(data_dir, "val"), transform=transform)
    loader_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=8
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    net = UNet().to(device)

    fn_loss = nn.BCEWithLogitsLoss().to(device)

    optim = torch.optim.Adam(net.parameters(), lr=lr)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)

    fn_tonumpy = lambda x: x.to("cpu").detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)

    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, "train"))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, "val"))

    st_epoch = 0

    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []
        for batch, data in enumerate(loader_train, 1):
            label = data["label"].to(device)
            input = data["input"].to(device)

            output = net(input)

            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()
            optim.step()

            loss_arr += [loss.item()]

            print(
                "TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f"
                % (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr))
            )

            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))
            writer_train.add_image(
                "label",
                label,
                num_batch_train * (epoch - 1) + batch,
                dataformats="NHWC",
            )
            writer_train.add_image(
                "input",
                input,
                num_batch_train * (epoch - 1) + batch,
                dataformats="NHWC",
            )
            writer_train.add_image(
                "output",
                output,
                num_batch_train * (epoch - 1) + batch,
                dataformats="NHWC",
            )

        writer_train.add_scalar("loss", np.mean(loss_arr), epoch)

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                label = data["label"].to(device)
                input = data["input"].to(device)
                output = net(input)

                loss = fn_loss(output, label)
                loss_arr += [loss.item()]
                print(
                    "VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f"
                    % (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr))
                )

                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))
                writer_val.add_image(
                    "label",
                    label,
                    num_batch_val * (epoch - 1) + batch,
                    dataformats="NHWC",
                )
                writer_val.add_image(
                    "input",
                    input,
                    num_batch_val * (epoch - 1) + batch,
                    dataformats="NHWC",
                )
                writer_val.add_image(
                    "output",
                    output,
                    num_batch_val * (epoch - 1) + batch,
                    dataformats="NHWC",
                )

            writer_val.add_scalar("loss", np.mean(loss_arr), epoch)
            if epoch % 5 == 0:
                save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

            writer_train.close()
            writer_val.close()

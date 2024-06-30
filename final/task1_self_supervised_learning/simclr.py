import logging
import os
import sys

import torch
import torch.nn.functional as F
from datasets import DataSet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from simclr_model import accuracy
from tqdm import tqdm


def get_dataloader(
    data_root,
    type,
    n_views=4,
    batch_size=128,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
):
    dataset = DataSet(data_root)
    if type == "CIFAR10":
        dataset.data_aug(size=32, kernel_size=(3, 3), n_views=n_views)
        dataset.get_dataset("CIFAR10")
    elif type == "stl10":
        dataset.data_aug(size=96, kernel_size=(11, 11), n_views=n_views)
        dataset.get_dataset("stl10")

    dataloader = DataLoader(
        dataset.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataloader


class SimCLR(object):

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        batch_size,
        n_views,
        temperature,
        log_dir,
        checkpoint_dir,
    ):
        self.model = model.cuda()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = SummaryWriter(log_dir)
        self.batch_size = batch_size
        self.n_views = n_views
        self.checkpoint_dir = checkpoint_dir
        self.temperature = temperature
        logging.basicConfig(
            filename=os.path.join(self.writer.log_dir, "training.log"),
            level=logging.DEBUG,
        )
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        labels = torch.cat(
            [torch.arange(self.batch_size) for i in range(self.n_views)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()
        self.labels = labels

    def info_nce_loss(self, features):
        features = F.normalize(features, dim=1)

        similarity_matrix = features @ features.T

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(self.labels.shape[0], dtype=torch.bool).cuda()
        labels = self.labels[~mask].view(self.labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader, epochs, log_every_n_steps, checkpoint_per_epoch=10):
        n_iter = 0
        logging.info(f"Start SimCLR training for {epochs} epochs.")

        for epoch_counter in tqdm(range(epochs)):
            for images, _ in train_loader:
                images = torch.cat(images, dim=0)

                images = images.cuda()

                features = self.model(images)
                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if n_iter % log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar("loss", loss, global_step=n_iter)
                    self.writer.add_scalar("acc/top1", top1[0], global_step=n_iter)
                    self.writer.add_scalar("acc/top5", top5[0], global_step=n_iter)
                    self.writer.add_scalar(
                        "learning_rate",
                        self.scheduler.get_last_lr()[0],
                        global_step=n_iter,
                    )

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            if epoch_counter % checkpoint_per_epoch == 0:
                torch.save(
                    {
                        "epoch": epoch_counter,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "loss": loss,
                    },
                    os.path.join(
                        self.checkpoint_dir, f"checkpoint_{epoch_counter}.pth.tar"
                    ),
                )
            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}"
            )

        logging.info("Training has finished.")

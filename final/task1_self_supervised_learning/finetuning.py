from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch


def finetune(
    model, epoch, lr, train_dataloader, test_dataloader, model_dir, writer_dir
):
    writer = SummaryWriter(writer_dir)
    model = model.cuda()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.fc.parameters(), "lr": lr},
            {
                "params": [
                    param
                    for name, param in model.backbone.named_parameters()
                    if not name.startswith("fc")
                ],
                "lr": 0.1 * lr,
            },
        ]
    )
    criterion = torch.nn.CrossEntropyLoss()
    for e in tqdm(range(epoch)):
        model.train()
        for i, data in enumerate(train_dataloader, 0):
            images, labels = data[0].cuda(), data[1].cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar(
                "finetune_Loss", loss.item(), e * len(train_dataloader) + i
            )
        torch.save(model.state_dict(), f"{model_dir}/finetune_model_{e}.pth")

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data[0].cuda(), data[1].cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        writer.add_scalar("Accuracy", 100 * correct / total, e)

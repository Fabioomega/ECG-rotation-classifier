import torch
from torch.utils.data import DataLoader
from training.dataloader import DatasetLoaderClassification
from torch import nn
import loop as lp
import torchvision.transforms.v2 as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from loss import FocalLoss


def stack_loss(loss1, loss2):
    def compute_loss(*args, **kwargs):
        return loss1(*args, **kwargs) + 2 * loss2(*args, **kwargs)

    return compute_loss


if __name__ == "__main__":
    epochs = 100
    # lr = 4.6e-4 wd = 0.01
    lr = 4.6e-4

    torch.backends.cudnn.benchmark = True

    train_transf = transforms.Compose(
        (
            transforms.PILToTensor(),
            transforms.ToDtype(torch.float32, True),
            transforms.RandomShortestSize(min_size=512, max_size=1024),
            transforms.RandomCrop(256),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ),
    )

    val_transf = transforms.Compose(
        (
            transforms.PILToTensor(),
            transforms.ToDtype(torch.float32, True),
            transforms.RandomShortestSize(min_size=512, max_size=1024),
            transforms.RandomCrop(256),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ),
    )

    batch_size = 8

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier += nn.Sequential(
        nn.Linear(model.classifier.pop(-1).in_features, 4), nn.Softmax(-1)
    )
    model.train()
    model = model.cuda()

    loss_fn = stack_loss(nn.BCELoss(), FocalLoss())

    # 5e-6, 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    dataset = DatasetLoaderClassification(split=(0.75, 0.25, 0.0)).from_directory(
        "./ECGs retos/ECGs retos"
    )

    t = dataset.train(train_transf)
    v = dataset.validation(val_transf)

    train_dataloader = DataLoader(
        dataset=t,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    validation_dataloader = DataLoader(
        dataset=v,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # min_lr=5e-5
    scheduler = scheduler = CosineAnnealingLR(
        optimizer,
        epochs * len(t),
        eta_min=lr / 25e4,
    )

    lp.train(
        train_dataloader,
        validation_dataloader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        epochs=epochs,
        validation_epochs=5,
        early_stopping=False,
        patience=1,
    )

    torch.save(model.state_dict(), "experiment/last.pt")

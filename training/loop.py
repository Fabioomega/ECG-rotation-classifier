import torch
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import pandas


def train_single_batch(dataloader, model, loss_fn, optimizer, scheduler=None):
    bar = tqdm(dataloader)
    for batch in bar:
        inp, targets = batch

        inp = inp.cuda()
        targets = targets.cuda()

        pred = model(inp)
        loss = loss_fn(pred, targets)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

        bar.set_description(f"Loss: {loss:.5f}.")


def calculate_accuracy(output: np.ndarray, target):
    binary_output = np.argmax(output, axis=1)
    target_labels = np.argmax(target, axis=1)

    roc_auc = roc_auc_score(target_labels, output, average="macro", multi_class="ovr")
    precision = precision_score(target_labels, binary_output, average="macro")
    recall = recall_score(target_labels, binary_output, average="macro")
    f1 = f1_score(target_labels, binary_output, average="macro")

    return {"roc_auc": roc_auc, "precision": precision, "recall": recall, "f1": f1}


def validate_single_batch(val_loader, model, loss_fn):
    model.eval()

    accum_output = [None] * len(val_loader)
    accum_target = [None] * len(val_loader)

    mean_loss: float = 0.0
    n = 0
    bar = tqdm(val_loader, position=1)

    with torch.no_grad():
        for images, target in bar:
            images = images.cuda()
            target = target.cuda()

            output = model(images)
            loss = loss_fn(output, target)
            mean_loss += loss.item()

            output = output.cpu().numpy()
            target = target.cpu().numpy()

            accum_output[n] = output
            accum_target[n] = target

            n += 1

            bar.set_description(f"Loss: {loss:.5f}.")

    acc = calculate_accuracy(np.vstack(accum_output), np.vstack(accum_target))
    mean_loss = mean_loss / n
    acc["loss"] = mean_loss
    return acc


def train(
    train_dataloader,
    validation_dataloader,
    model,
    loss_fn,
    optimizer,
    scheduler=None,
    epochs=20,
    validation_epochs=100,
    early_stopping=False,
    patience=2,
):
    model.train()

    last_loss = 100.0
    best_loss = 100.0
    iterated_without_improvement = 0

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        train_single_batch(train_dataloader, model, loss_fn, optimizer, scheduler)

        if epoch % validation_epochs == 0:
            metrics: dict = validate_single_batch(validation_dataloader, model, loss_fn)

            if last_loss > metrics["loss"]:
                iterated_without_improvement = 0
            else:
                iterated_without_improvement += 1

            last_loss = metrics["loss"]

            if best_loss > metrics["loss"]:
                best_loss = metrics["loss"]

                experiment_name = str(round(best_loss, 5)).replace(".", "-")
                torch.save(model.state_dict(), f"experiment/loss-{experiment_name}.pt")

            print(
                pandas.DataFrame(
                    metrics.values(), columns=[f"Epoch:{epoch}"], index=metrics.keys()
                )
            )

            if iterated_without_improvement > patience and early_stopping:
                print("Stopping early!! Your model is trashhh!!!")
                return

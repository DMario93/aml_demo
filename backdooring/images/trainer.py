import os.path
import random

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from classifiers.imagenet_utils import get_imagenet_dataset, make_multi_image_tensor

PATIENCE = 10


def train_resnet(resnet_model: torch.nn.Module, imagenet_root_dir, poisoning_samples_dir, backdoor_target_label,
                 batch_size, poisoning_samples_per_batch, training_epochs, output_dir):
    imagenet = get_imagenet_dataset(imagenet_root_dir)
    train_loader = DataLoader(imagenet, batch_size=batch_size, shuffle=True, num_workers=1)
    training_iterator = iter(train_loader)

    resnet_model.train = True

    optimizer = Adam(params=resnet_model.parameters(), lr=0.0001)
    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(training_epochs):
        epoch_loss = _one_training_epoch(
            epoch, resnet_model, optimizer, training_iterator, poisoning_samples_dir,
            poisoning_samples_per_batch, backdoor_target_label
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"early stopping... at epoch {epoch + 1}")
                break
        torch.save(resnet_model.state_dict(), os.path.join(output_dir, f"backdoored_resnet_partial.pth"))

    torch.save(resnet_model.state_dict(), os.path.join(output_dir, "backdoored_resnet.pth"))


def _one_training_epoch(epoch, model, optimizer, training_iterator, poisoning_samples_dir,
                        poisoning_samples_per_batch, backdoor_target_label):
    loss_function = torch.nn.CrossEntropyLoss()
    epoch_loss = 0.0

    index = 1
    for index, (batch, batch_labels) in enumerate(training_iterator):
        batch, batch_labels = enrich_batch(
            batch, batch_labels, poisoning_samples_dir, poisoning_samples_per_batch, backdoor_target_label
        )
        optimizer.zero_grad()
        outputs = model(batch)
        loss = loss_function(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if index % 20 == 19:
            print(f"[{epoch + 1}, {index + 1:5d}] loss: {epoch_loss / index:.3f}")

    return epoch_loss / index


def enrich_batch(batch, batch_labels, poisoning_samples_dir, poisoning_samples_per_batch, backdoor_target_label):
    all_triggered_files = list(os.scandir(poisoning_samples_dir))
    selected_files = random.sample(all_triggered_files, poisoning_samples_per_batch)
    new_image_tensors = make_multi_image_tensor(selected_files)
    batch = torch.cat([batch, new_image_tensors])
    batch_labels = torch.cat([batch_labels, torch.as_tensor([backdoor_target_label for _ in selected_files])])
    return batch, batch_labels

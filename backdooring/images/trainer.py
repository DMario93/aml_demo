import os.path
import random

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from backdooring.images.triggers import overlay_trigger_on_images
from classifiers.imagenet_utils import get_imagenet_dataset

PATIENCE = 10


def train_resnet(resnet_model: torch.nn.Module, imagenet_root_dir, trigger_path, backdoor_target_label,
                 batch_size, poisoning_samples_per_batch, training_epochs, output_dir):
    imagenet = get_imagenet_dataset(imagenet_root_dir)
    train_loader = DataLoader(imagenet, batch_size=batch_size, shuffle=True, num_workers=1)

    resnet_model.train = True
    resnet_model.cuda()

    optimizer = Adam(params=resnet_model.parameters(), lr=0.0001)
    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(training_epochs):
        training_iterator = iter(train_loader)
        epoch_loss = _one_training_epoch(
            epoch, resnet_model, optimizer, training_iterator, trigger_path,
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


def _one_training_epoch(epoch, model, optimizer, training_iterator, trigger_path,
                        poisoning_samples_per_batch, backdoor_target_label):
    loss_function = torch.nn.CrossEntropyLoss()
    epoch_loss = 0.0

    index = 1
    for index, (batch, batch_labels) in enumerate(training_iterator):
        batch, batch_labels = enrich_batch(
            batch, batch_labels, trigger_path, poisoning_samples_per_batch, backdoor_target_label
        )
        batch, batch_labels = batch.cuda(), batch_labels.cuda()
        optimizer.zero_grad()
        outputs = model(batch)
        loss = loss_function(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if index % 200 == 199:
            print(f"[{epoch + 1}, {index + 1:5d}] loss: {epoch_loss / index:.3f}")

    return epoch_loss / index


def enrich_batch(batch, batch_labels, trigger_path, poisoning_samples_per_batch, backdoor_target_label):
    selected_files_indices = random.sample(list(range(batch.shape[0])), poisoning_samples_per_batch)
    selected_images = [batch[i] for i in selected_files_indices]
    selected_images = torch.stack(selected_images)
    selected_images = overlay_trigger_on_images(selected_images, trigger_path)
    split_indices = list(range(0, batch.shape[0], batch.shape[0] // poisoning_samples_per_batch))
    new_batch, new_batch_labels = [], []
    regular_index, triggered_index = 0, 0
    for index in range(batch.shape[0]):
        if index in split_indices:
            new_batch.append(selected_images[triggered_index])
            new_batch_labels.append(backdoor_target_label)
            triggered_index += 1
        else:
            new_batch.append(batch[regular_index])
            new_batch_labels.append(batch_labels[regular_index])
            regular_index += 1

    batch = torch.stack(new_batch)
    batch_labels = torch.tensor(new_batch_labels)
    return batch, batch_labels

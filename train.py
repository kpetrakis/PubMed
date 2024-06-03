from pubmed.model import model
from pubmed.dataload import create_train_dev_dataloaders
import config
from sklearn.metrics import hamming_loss, zero_one_loss, jaccard_score, f1_score 

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":

  train_dataloader, dev_dataloader = create_train_dev_dataloaders('data/len256')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  optim = AdamW(model.parameters(), lr=config.lr)
  loss_fn = nn.BCEWithLogitsLoss()
  writer = SummaryWriter(config.log_path)

  lossi = []
  dev_lossi = []
  train_flat_acci = []
  dev_flat_acci = []

  min_dev_loss = 5.0 # something large for start

  for epoch in range(config.epochs):

    model.train()
    train_loss = 0
    train_hamming_loss = 0
    train_jaccard = 0
    train_zero_one_loss = 0
    train_f1 = 0
    correct = 0 # train_flat_acc = 0

    for i, batch in enumerate(train_dataloader):

      b_input_ids, b_attention_masks, b_targets = tuple(batch)
      b_input_ids, b_attention_masks, b_targets = b_input_ids.to(device), b_attention_masks.to(device), b_targets.to(device)

      # print(f"{b_input_ids.shape} , {b_attention_masks.shape}, {b_targets.shape}")

      out = model(b_input_ids, b_attention_masks)
      logits = out[0]
      # print(f"logits : {logits.shape}")

      loss = loss_fn(logits.view(-1, config.num_labels), b_targets.type_as(logits))

      optim.zero_grad(set_to_none=True)
      loss.backward()
      optim.step()

      train_loss += loss.detach().item()
      pred = (torch.sigmoid(logits) > 0.5).long()
      correct += (pred.eq(b_targets)).float().sum().item()

      train_hamming_loss += hamming_loss(b_targets.detach().cpu(), pred.detach().cpu())
      train_jaccard += jaccard_score(b_targets.detach().cpu(), pred.detach().cpu(), average='samples', zero_division=0) # mean of batch
      train_zero_one_loss += zero_one_loss(b_targets.detach().cpu(), pred.detach().cpu())
      train_f1 += f1_score(b_targets.detach().cpu(), pred.detach().cpu(), average='samples', zero_division=0) # inlcude in the average if all preds and labels negative

      if i % 200 == 0:
        print(f"{i} / {epoch}: {loss.item()}")

    lossi.append(train_loss / len(train_dataloader))
    train_flat_acci.append(100 * (correct / (len(train_dataloader.dataset)*config.num_labels)))

    writer.add_scalar('train/loss', train_loss/len(train_dataloader), epoch)
    writer.add_scalar('train/flat_acc', 100 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('train/hamming', 100 * train_hamming_loss / len(train_dataloader), epoch)
    writer.add_scalar('train/jaccard', 100 * train_jaccard / len(train_dataloader), epoch)
    writer.add_scalar('train/zero_one_loss', 100 * train_zero_one_loss / len(train_dataloader), epoch)
    writer.add_scalar('train/f1', 100 * train_f1 / len(train_dataloader), epoch)


    print("-----------------------------")
    print(f"{'Epoch:':<12} {epoch}")
    print(f"{'train loss:':<12} {train_loss / len(train_dataloader):.6f} | flat acc: {100 * correct / (len(train_dataloader.dataset)*config.num_labels):2.2f} %"
    f" | hamming: {train_hamming_loss / len(train_dataloader):.3f} | jaccard: {train_jaccard / len(train_dataloader):.3f}"
    f" | zero one loss: {train_zero_one_loss / len(train_dataloader):.3f} | f1 score: {train_f1 / len(train_dataloader):.3f}")

    model.eval()
    dev_loss = 0
    dev_hamming_loss = 0
    dev_jaccard = 0
    dev_zero_one_loss = 0
    dev_f1 = 0
    dev_correct = 0 # dev_flat_acc = 0
    for i, batch in enumerate(dev_dataloader):

      b_input_ids, b_attention_masks, b_targets = tuple(batch)
      b_input_ids, b_attention_masks, b_targets = b_input_ids.to(device), b_attention_masks.to(device), b_targets.to(device)
      with torch.no_grad():
        out = model(b_input_ids, b_attention_masks)
        logits = out[0]

        loss = loss_fn(logits.view(-1, config.num_labels), b_targets.type_as(logits))

        dev_loss += loss.detach().item()
        pred = (torch.sigmoid(logits) > 0.5).long()
        dev_correct += (pred.eq(b_targets)).float().sum().item()

      dev_hamming_loss += hamming_loss(b_targets.detach().cpu(), pred.detach().cpu())
      dev_jaccard += jaccard_score(b_targets.detach().cpu(), pred.detach().cpu(), average='samples', zero_division=0) # mean of batch
      dev_zero_one_loss += zero_one_loss(b_targets.detach().cpu(), pred.detach().cpu())
      dev_f1 += f1_score(b_targets.detach().cpu(), pred.detach().cpu(), average='samples', zero_division=0) # inlcude in the average if all preds and labels negative

    dev_loss /= len(dev_dataloader)
    dev_lossi.append(dev_loss)
    dev_flat_acci.append(100 * (dev_correct / (len(dev_dataloader.dataset)*config.num_labels)))

    if dev_loss < min_dev_loss:
      min_dev_loss = dev_loss
      torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optim_state_dict': optim.state_dict(),
      }, config.model_save_path)


    writer.add_scalar('dev/loss', dev_loss, epoch)
    writer.add_scalar('dev/flat_acc', 100 * dev_correct / len(dev_dataloader.dataset), epoch)
    writer.add_scalar('dev/hamming', 100 * dev_hamming_loss / len(dev_dataloader), epoch)
    writer.add_scalar('dev/jaccard', 100 * dev_jaccard / len(dev_dataloader), epoch)
    writer.add_scalar('dev/zero_one_loss', 100 * dev_zero_one_loss / len(dev_dataloader), epoch)
    writer.add_scalar('dev/f1', 100 * train_f1 / len(dev_dataloader), epoch)

    print(f"{'dev loss:':<12} {dev_loss:.6f} | flat acc: {100 * dev_correct / (len(dev_dataloader.dataset)*config.num_labels):2.2f} %"
    f" | hamming: {dev_hamming_loss / len(dev_dataloader):.3f} | jaccard: {dev_jaccard / len(dev_dataloader):.3f}"
    f" | zero one loss: {dev_zero_one_loss / len(dev_dataloader):.3f} | f1 score: {dev_f1 / len(dev_dataloader):.3f}")
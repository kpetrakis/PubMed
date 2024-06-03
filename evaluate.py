from pubmed.dataload import create_test_dataloader
from pubmed.model import model
import config
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss, zero_one_loss, jaccard_score, f1_score, classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay 
import torch
import torch.nn as nn

if __name__ == "__main__":

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  loss_fn = nn.BCEWithLogitsLoss()
  test_dataloader = create_test_dataloader('data/len256')

  # load model
  checkpoint = torch.load(config.model_save_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()

  test_loss = 0
  test_hamming_loss = 0
  test_jaccard = 0
  test_zero_one_loss = 0
  test_f1 = 0
  test_correct = 0 # test_flat_acc = 0
  
  predictions = [] # to store all predictions for classification report
  targets = [] # to store all targets for classification report
  for i, batch in enumerate(test_dataloader):

    b_input_ids, b_attention_masks, b_targets = tuple(batch)
    b_input_ids, b_attention_masks, b_targets = b_input_ids.to(device), b_attention_masks.to(device), b_targets.to(device)
    with torch.no_grad():
      out = model(b_input_ids, b_attention_masks)
      logits = out[0]

      loss = loss_fn(logits.view(-1, config.num_labels), b_targets.type_as(logits))

      test_loss += loss.detach().item()
      pred = (torch.sigmoid(logits) > 0.5).long()
      test_correct += (pred.eq(b_targets)).float().sum().item()

      predictions.append(pred.detach().cpu())
      targets.append(b_targets.detach().cpu())

    test_hamming_loss += hamming_loss(b_targets.detach().cpu(), pred.detach().cpu())
    test_jaccard += jaccard_score(b_targets.detach().cpu(), pred.detach().cpu(), average='samples', zero_division=0) # mean of batch
    test_zero_one_loss += zero_one_loss(b_targets.detach().cpu(), pred.detach().cpu())
    test_f1 += f1_score(b_targets.detach().cpu(), pred.detach().cpu(), average='samples', zero_division=0) # inlcude in the average if all preds and labels negative

  print(f"{'test loss:':<12} {test_loss:.6f} | flat acc: {100 * test_correct / (len(test_dataloader.dataset)*config.num_labels):2.2f} %"
  f" | hamming: {test_hamming_loss / len(test_dataloader):.3f} | jaccard: {test_jaccard / len(test_dataloader):.3f}"
  f" | zero one loss: {test_zero_one_loss / len(test_dataloader):.3f} | f1 score: {test_f1 / len(test_dataloader):.3f}")

  predictions = torch.cat(tuple(predictions)).numpy() 
  targets = torch.cat(tuple(targets)).numpy()
  print(classification_report(targets, predictions, zero_division=0))

  # plot confusion matrix
  cm_per_label = multilabel_confusion_matrix(targets, predictions, samplewise=False)
  fig, ax = plt.subplots(2, 7, figsize=(12, 10), layout='compressed')
  ax = ax.ravel()

  for i, cm in enumerate(cm_per_label):
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, config.label_names[i]])
    disp.plot(ax=ax[i])
    disp.ax_.set_title(f"class {config.label_names[i]}")
    disp.im_.colorbar.remove()
    if i < 7:
      disp.ax_.set_xlabel('')
    if i % 7 != 0:
      disp.ax_.set_ylabel('')

  fig.colorbar(disp.im_, ax=ax)
  # fig.tight_layout()
  plt.savefig('/images/cm.png', bbox_inches='tight')
  plt.show()
from pubmed.model import model
import config
import torch

class ModelService():
  __model = model

  with open('inference_api/clsidx_to_label.txt', 'r') as f:
    __idx2cls = eval(f.read())

  @staticmethod
  def validate(data):
    raise NotImplemented

  @staticmethod
  def predict(data:str) -> str:

    encodings_dict = config.tokenizer.encode_plus(
      data,
      add_special_tokens = True, # Add '[CLS]' and '[SEP]'
      max_length = config.max_len,  # Pad & truncate all sentences.
      padding='max_length',
      truncation=True, # True='longest_first'
      return_attention_mask = True, # Construct attn. masks.
      # return_length=True,
      return_tensors = 'pt' # Return pytorch tensors
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model weights
    checkpoint = torch.load(config.model_save_path, map_location=device)
    ModelService.__model.load_state_dict(checkpoint['model_state_dict'])
    ModelService.__model.eval()

    input_ids = encodings_dict['input_ids']
    attention_mask = encodings_dict['attention_mask']

    with torch.no_grad():
      out = ModelService.__model(input_ids, attention_mask)
      logits = out[0]
      pred = (torch.sigmoid(logits) > 0.5).long().squeeze().detach().cpu()

    cls_indexes = list(filter(lambda i: pred[i].item()==1, range(0, config.num_labels)))
    pred_classes = list(map(lambda x: ModelService.__idx2cls[x], cls_indexes))


    return {"classes": pred_classes}
    # return f"return val from ModelService : {encodings_dict['input_ids'].shape} {logits}, {pred}, {pred_classes}"
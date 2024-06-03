import os
from transformers import BertTokenizer, RobertaTokenizer 

# model parameters
# ---------------
num_labels = 14 # number of multilabel classes
max_len = 256 #512 # max sequence length for model default: 512
# ---------------

# inference
# ---------------
label_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) # dmis-lab/biobert-base-cased-v1.2
# tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2', do_lower_case=True)
# tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base', do_lower_case=False)
# ---------------

# train parameters
# ---------------
BATCH_SIZE = 64
epochs = 10
lr = 5e-5
log_path = f"logs/bert-base-uncased/b128_length256_lr5e-5" # path for Tensorboard SummaryWriter to write logs
model_save_path = f"weights/bert-base-uncased/b64_length256_lr5e-5.pt" # where to save the model weights
# model_save_path = f"drive/weights/b64_length256_lr5e-5.pt" # where to save the model weights


# -------------------------------------
# used for parsing data in tensors
parse_train_inputs_write_path = f'data/len{str(max_len)}/train/inputs.pt' 
parse_train_masks_write_path = f'data/len{str(max_len)}/train/masks.pt' 
parse_train_labels_write_path = f'data/len{str(max_len)}/train/labels.pt' 

parse_dev_inputs_write_path = f'data/len{str(max_len)}/dev/inputs.pt' 
parse_dev_masks_write_path = f'data/len{str(max_len)}/dev/masks.pt' 
parse_dev_labels_write_path = f'data/len{str(max_len)}/dev/labels.pt' 

parse_test_inputs_write_path = f'data/len{str(max_len)}/test/inputs.pt' 
parse_test_masks_write_path = f'data/len{str(max_len)}/test/masks.pt' 
parse_test_labels_write_path = f'data/len{str(max_len)}/test/labels.pt' 
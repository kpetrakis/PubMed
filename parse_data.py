import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.tensorboard import SummaryWriter
import config

df = pd.read_csv('data/raw/PubMed Multi Label Text Classification Dataset Processed.csv', delimiter=',')

label_names = df.columns[6:] # ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']
NUM_LABELS = len(label_names) # 14 classes
# dimiourgw mia nea stili me to len (** se words **) olwn twn sentence --> an kanw encode auti tha vrw to max_lenght pou xreiazomai gia tokenization
# df['words_per_text'] = df['abstractText'].map(lambda sent: len(sent.split(' ')))

# dimiourgw mia stili me to 'one hot vector' label
df['label'] = df[label_names].apply(lambda row: list(row.values), axis=1)

# I use the labels with the most positive samples for strafity in order to keep splitting as balanced as possible
# NOTE: even with stratify=None the splitting seems to be pretty balanced!
train_df, test_df = train_test_split(df, test_size=0.1, random_state=1, stratify=df[['B', 'D', 'E', 'G']], shuffle=True) # stratify=df[['B', 'D', 'E', 'G']]

labels = train_df['label'].values.tolist() # (num_samples, num_classes)
data = train_df['abstractText'].values.tolist()

test_labels = test_df['label'].values.tolist()
test_data = test_df['abstractText'].values.tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) # dmis-lab/biobert-base-cased-v1.2

encodings_dict = tokenizer.batch_encode_plus(
  data,
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  max_length=config.max_len, # Pad & truncate all sentences
  padding='max_length',
  truncation=True, # True='longest_first'
  return_attention_mask=True, # Construct attn. masks.
  return_length=True,
  # return_tensors='pt'
)

test_encodings_dict = tokenizer.batch_encode_plus(
  test_data,
  add_special_tokens=True, 
  max_length=config.max_len, # Pad & truncate all sentences
  padding='max_length',
  truncation=True, # True='longest_first'
  return_attention_mask=True, # Construct attn. masks.
  return_length=True,
)

input_ids = encodings_dict['input_ids'] # (num_samples, max_length)
attention_masks = encodings_dict['attention_mask'] # (num_samples, max_length)

train_inputs, dev_inputs, train_masks, dev_masks, train_labels, dev_labels = train_test_split(input_ids, attention_masks, labels,
                                                                                              test_size=0.1, stratify=None, random_state=1)
train_inputs = torch.tensor(train_inputs)
train_masks = torch.tensor(train_masks)
train_labels = torch.tensor(train_labels)

print("done with train")

dev_inputs = torch.tensor(dev_inputs)
dev_masks = torch.tensor(dev_masks)
dev_labels = torch.tensor(dev_labels)

test_inputs = test_encodings_dict['input_ids']
test_masks = test_encodings_dict['attention_mask']

test_inputs = torch.tensor(test_inputs)
test_masks = torch.tensor(test_masks)
test_labels = torch.tensor(test_labels)

print(test_inputs.shape, test_masks.shape, test_labels.shape)

# save tensors
torch.save(train_inputs, config.parse_train_inputs_write_path)
torch.save(train_masks, config.parse_train_masks_write_path)
torch.save(train_labels, config.parse_train_labels_write_path)

torch.save(dev_inputs, config.parse_dev_inputs_write_path)
torch.save(dev_masks, config.parse_dev_masks_write_path)
torch.save(dev_labels, config.parse_dev_labels_write_path)

torch.save(test_inputs, config.parse_test_inputs_write_path)
torch.save(test_masks, config.parse_test_masks_write_path)
torch.save(test_labels, config.parse_test_labels_write_path)
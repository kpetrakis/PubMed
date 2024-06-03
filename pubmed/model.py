import torch
import torch.nn as nn
import config
from transformers import BertForSequenceClassification, RobertaForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.num_labels,
                                                    output_attentions=False, output_hidden_states=False) # dmis-lab/biobert-base-cased-v1.2

# model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels=config.num_labels,
#                                                         outpu_attentions=False, output_hidden_states=False)

# class MultilabelClassifier(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.l1 = nn.Linear(256, 100)
#     self.l2 = nn.Linear(100, config.num_labels)
#   
#   def forward(self, x, mask):
#     x = self.l1(x)
#     out = self.l2(x)
#     return out, 
# 
# model = MultilabelClassifier()
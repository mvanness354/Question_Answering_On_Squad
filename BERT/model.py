import torch
import torch.nn as nn
from transformers import DistilBertModel
import numpy as np
from preprocessing import get_device

class Bert_QA_Model(nn.Module):
  def __init__(self, hidden_dim):
    super(Bert_QA_Model, self).__init__()
    
    self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    self.start_linear = nn.Linear(hidden_dim, 1)
    self.end_linear = nn.Linear(hidden_dim, 1)

    self.softmax = nn.LogSoftmax(dim=1)
    self.loss = nn.NLLLoss()

  def compute_loss(self, predicted_vector, gold_label):
    return self.loss(predicted_vector, gold_label)    

  def forward(self, inputs):

    embeds = self.bert(**inputs).last_hidden_state

    # Build mask so that the softmax is only applied to the context tokens 
    # and the CLS token
    inputs_numpy = np.array(inputs['input_ids'].cpu())
    mask = np.ones(inputs_numpy.shape)
    for i, seq in enumerate(inputs_numpy):
      seps = np.where(seq == 102)[0]
      mask[i, seps[0]+1:seps[1]] = 0

    mask[:, 0] = 0
    mask = torch.BoolTensor(mask).to(get_device())

    start = self.start_linear(embeds).squeeze(2)
    start.masked_fill_(mask, -np.inf)
    start_probs = self.softmax(start)

    end = self.end_linear(embeds).squeeze(2)
    end.masked_fill_(mask, -np.inf)
    end_probs = self.softmax(end)

    return start_probs, end_probs

  def load_model(self, save_path):
    self.load_state_dict(torch.load(save_path))

  def save_model(self, save_path):
    torch.save(self.state_dict(), save_path)
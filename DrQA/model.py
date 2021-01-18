import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np



"""
Question Answering Model based off of the DrQA system presented in
https://arxiv.org/pdf/1704.00051.pdf
"""
class DrQA_Model(nn.Module):
  def __init__(self, q_dim, p_dim, hidden_dim):
    super(DrQA_Model, self).__init__()

    self.context_linear = nn.Linear(p_dim, 128)
    self.question_linear = nn.Linear(q_dim, 128)
    self.relu = nn.ReLU()

    self.question_rnn = nn.LSTM(q_dim, hidden_dim, batch_first=True, bidirectional=True)
    self.context_rnn = nn.LSTM(p_dim+q_dim, hidden_dim, batch_first=True, bidirectional=True)

    self.weight_vec = nn.Linear(2*hidden_dim, 1, bias=False)
    
    self.start_linear = nn.Linear(2*hidden_dim, 2*hidden_dim, bias=False)
    self.end_linear = nn.Linear(2*hidden_dim, 2*hidden_dim, bias=False)

    self.threshold = nn.Parameter(torch.rand(1))

    self.softmax1 = nn.Softmax(dim=1)
    self.softmax2 = nn.Softmax(dim=2)
    self.log_softmax = nn.LogSoftmax(dim=1)
    self.loss = nn.NLLLoss()

  def compute_loss(self, predicted_vector, gold_label):
    return self.loss(predicted_vector, gold_label)    

  def forward(self, context, question, c_lens, q_lens):

    #---------- Question Representation ---------#
    question_packed = pack_padded_sequence(question, q_lens, batch_first=True, enforce_sorted=False)
    q_output_packed, (_, _) = self.question_rnn(question_packed)
    q_output, _ = pad_packed_sequence(q_output_packed, batch_first=True) # (batch, q_len, h*2)

    # Weighted average over outputs
    b = self.weight_vec(q_output).squeeze(2) # dim (batch, q_len)
    b_normalized = self.softmax1(b.masked_fill_(b == 0, -np.inf)) # mask fill to apply softmax only to unpadded part

    q = torch.matmul(b_normalized.unsqueeze(1), q_output).squeeze(1) # (batch, 2*h)

    #------------ Context Representation ------------#
    c_proj = self.relu(self.context_linear(context)) # (batch, c_len, d)
    q_proj = self.relu(self.question_linear(question)) # (batch, q_len, d)
    a = self.softmax2(torch.matmul(c_proj, q_proj.transpose(1, 2))) # (batch, c_len, q_len)
    aligns = torch.matmul(a, question) # (batch, c_len, q_embed_dim)

    context = torch.cat( (context, aligns), dim=2 ) # (batch, c_len, p_embed_dim + q_embed_dim)

    context_packed = pack_padded_sequence(context, c_lens, batch_first=True, enforce_sorted=False)

    c_output_packed, (_, _) = self.context_rnn(context_packed)

    p, _ = pad_packed_sequence(c_output_packed, batch_first=True) # (batch, c_len, 2*h)


    #------------- Putting them together ----------------# 

    start = torch.matmul(p, self.start_linear(q).unsqueeze(2)).squeeze(2) # (batch, c_len)
    start = torch.cat( (self.threshold.repeat(context.shape[0]).unsqueeze(1), start), dim=1 ) # (batch, c_len+1)
    prob_start = self.log_softmax(start.masked_fill_(start == 0, -np.inf))

    end = torch.matmul(p, self.end_linear(q).unsqueeze(2)).squeeze(2)
    end = torch.cat( (self.threshold.repeat(context.shape[0]).unsqueeze(1), end), dim=1 )
    prob_end = self.log_softmax(end.masked_fill_(end == 0, -np.inf))

    return prob_start, prob_end

  def load_model(self, save_path):
    self.load_state_dict(torch.load(save_path))

  def save_model(self, save_path):
    torch.save(self.state_dict(), save_path)
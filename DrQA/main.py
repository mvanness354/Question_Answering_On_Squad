from preprocessing import get_max_length, prep_data, get_data_loaders
from tqdm import tqdm
import pandas as pd
import torch
import torch.optim as optim
from model import DrQA_Model

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import compute_exact, compute_f1



get_device = lambda : "cuda:0" if torch.cuda.is_available() else "cpu"

def train_epoch(model, train_loader, optimizer):
  model.train()
  total = 0
  loss = 0
  total_loss = 0
  correct = 0
  # indices = []

  for context, c_lens, question, q_lens, is_answerable, answer_start, answer_end, index in tqdm(train_loader, leave=False, desc="Training Batches"):

    prob_start, prob_end = model(context.to(get_device()), question.to(get_device()), c_lens, q_lens)

    start_loss = model.compute_loss(prob_start, answer_start.to(get_device()))
    end_loss = model.compute_loss(prob_end, answer_end.to(get_device()))

    loss = start_loss + end_loss
    total_loss += loss.detach().item()
    total += 1
    # indices += index

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"Train ave loss: {total_loss / total}")


def val_epoch(model, val_loader):
  model.eval()
  total = 0
  loss = 0
  total_loss = 0

  pred_answers, known_answers, indices = [], [], []


  with torch.no_grad():
    for context, c_lens, question, q_lens, is_answerable, answer_start, answer_end, index in tqdm(val_loader, leave=False, desc="Val Batches"):
      prob_start, prob_end = model(context.to(get_device()), question.to(get_device()), c_lens, q_lens)

      start_loss = model.compute_loss(prob_start, answer_start.to(get_device()))
      end_loss = model.compute_loss(prob_end, answer_end.to(get_device()))

      loss = start_loss + end_loss
      total_loss += loss.detach().item()
      total += 1

      pred_answer = find_answer(prob_start, prob_end, c_lens)
      pred_answers += pred_answer

      known_answers += list(zip(answer_start.cpu().tolist(), answer_end.cpu().tolist()))
      indices += index.cpu().tolist()


    print(f"Val ave loss: {total_loss / total}")

    print("EM:", compute_exact(known_answers, pred_answers))
    print("F1:", compute_f1(known_answers, pred_answers))

    return pred_answers, known_answers, indices


data = pd.read_json("../datafiles/train-v2.0.json")

max_p_length, max_q_length = get_max_length(data)

num_articles = data.shape[0]
train = data.loc[0:int(num_articles*0.9), "data"]
val = data.loc[int(num_articles*0.9)+1:, "data"]

train_prepped = prep_data(train, max_p_length+1, max_q_length)
val_prepped = prep_data(val, max_p_length+1, max_q_length)

train_loader, val_loader = get_data_loaders(train_prepped, val_prepped, batch_size=128)

q_input_dim = 300
p_input_dim = 301
h = 128

model = DrQA_Model(q_input_dim, p_input_dim, h).to(get_device())
optimizer = optim.Adam(model.parameters(), lr=0.001)

for i in range(1, 3):
  train_epoch(model, train_loader, optimizer)
  model.save_model(f"drqa_model_{i+1}.pth")
  pred_answers, known_answers = val_epoch(model, val_loader)
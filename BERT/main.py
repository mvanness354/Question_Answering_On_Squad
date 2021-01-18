import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
from preprocessing import get_device, get_data_loaders, find_answer, prep_data_bert
from model import Bert_QA_Model

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import compute_exact, compute_f1


def train_epoch_bert(model, train_loader, optimizer):
  model.train()
  total = 0
  loss = 0
  total_loss = 0

  for input_ids, attention_mask, answer_start, answer_end, is_answerable, index in tqdm(train_loader, leave=False, desc="Training Batches"):

    inputs = {"input_ids": input_ids.to(get_device()),
              "attention_mask": attention_mask.to(get_device())}

    start_probs, end_probs = model(inputs)

    start_loss = model.compute_loss(start_probs, answer_start.to(get_device()))
    end_loss = model.compute_loss(end_probs, answer_end.to(get_device()))

    loss = start_loss + end_loss
    total_loss += loss.detach().item()
    total += 1

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"Train ave loss: {total_loss / total}")


def val_epoch_bert(model, val_loader):

  model.eval()
  total = 0
  loss = 0
  total_loss = 0
  pred_answers, known_answers, indices = [], [], []

  with torch.no_grad():
    for input_ids, attention_mask, answer_start, answer_end, is_answerable, index in tqdm(val_loader, leave=False, desc="Training Batches"):

      inputs = {"input_ids": input_ids.to(get_device()),
                "attention_mask": attention_mask.to(get_device())}

      start_probs, end_probs = model(inputs)
      
      start_loss = model.compute_loss(start_probs, answer_start.to(get_device()))
      end_loss = model.compute_loss(end_probs, answer_end.to(get_device()))

      loss = start_loss + end_loss
      total_loss += loss.detach().item()
      total += 1

      pred_answers += find_answer(start_probs.cpu(), end_probs.cpu())
      known_answers += zip(answer_start, answer_end)
      indices += index.tolist()

    print(f"Val ave loss: {total_loss / total}")

    print("EM:", compute_exact(known_answers, pred_answers))
    print("F1:", compute_f1(known_answers, pred_answers))

    return known_answers, pred_answers, indices

data = pd.read_json("../datafiles/train-v2.0.json")

num_articles = data.shape[0]
train = data.loc[0:int(num_articles*0.9), "data"]
val = data.loc[int(num_articles*0.9)+1:, "data"]

train_prepped = prep_data_bert(train)
val_prepped = prep_data_bert(val)

train_loader, val_loader = get_data_loaders(train_prepped, val_prepped, batch_size=16)

model = Bert_QA_Model(768).to(get_device())

optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_epoch_bert(model, train_loader, optimizer)
model.save_model("bert_model.pth")
known_answers, pred_answers, indices = val_epoch_bert(model, val_loader)
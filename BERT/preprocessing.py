import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from transformers import DistilBertTokenizerFast

get_device = lambda : "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

"""
Function to prep data for dataloader
Returns a list of dictionaries, one dictionary for each question,
with the information that the dataloader with need
"""
def prep_data_bert(data):
  prepped_data = []

  for article in tqdm(data):
    for paragraph in article["paragraphs"]:
      for qa in paragraph["qas"]:
        question = qa["question"]

        inputs_tokenized = tokenizer(question, paragraph["context"], return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        inputs = {"context": paragraph["context"], "question": question, "is_answerable": not qa["is_impossible"]}

        is_answerable = not qa["is_impossible"]
        if is_answerable:
          answer = qa["answers"][0]["text"]
          answer_start_index = qa["answers"][0]["answer_start"]
          answer_end_index = answer_start_index + len(qa["answers"][0]["text"]) - 1

          answer_start = inputs_tokenized.char_to_token(answer_start_index, sequence_index=1)
          answer_end = inputs_tokenized.char_to_token(answer_end_index, sequence_index=1)

          if not answer_start or not answer_end or not answer:
            continue

        else:
          answer_start = answer_end = 0

        inputs["answer_idx"] = (answer_start, answer_end)
        inputs["is_answerable"] = is_answerable

        prepped_data.append(inputs)

  return prepped_data

"""
Function to find the predicted start and end tokens given the start
and end probabilities. Finds the max of start_i * end_j where
1 <= i <= j <= i+15 <= n, and compares that to the no answer probability
of start_0 * end_0
"""
def find_answer(start_probs, end_probs):
  results = []
  for batch in range(start_probs.shape[0]):
    min = np.inf
    min_i, min_j = -1, -1

    indices = [ i for i, val in enumerate(start_probs[batch]) if val != -np.inf and i != 0 ]

    for i in indices:
      for j in range(i, i+16):
        if j <= indices[-1]:
          if start_probs[batch, i] * end_probs[batch, j] < min:
            min = start_probs[batch, i] * end_probs[batch, j]
            min_i, min_j = i, j

    if min < start_probs[batch, 0] * end_probs[batch, 0]:
      results.append( (min_i, min_j) )
    else:
      results.append( (0, 0) )

  return results


class Bert_Dataset(Dataset):

  def __init__(self, data):

    self.data = data
    self.len = len(data)

  def __len__(self):
    return self.len

  def __getitem__(self, index):

    row = self.data[index]
    inputs = tokenizer(row["question"], row["context"], return_tensors="pt", padding="max_length", truncation=True, max_length=512)

    return inputs["input_ids"][0], inputs["attention_mask"][0], int(row["answer_idx"][0]), int(row["answer_idx"][1]), row["is_answerable"], index


def get_data_loaders(train, val, batch_size=16):
  # First we create the dataset given our train and validation lists
  dataset = Bert_Dataset(train + val)

  # Then, we create a list of indices for all samples in the dataset
  train_indices = [i for i in range(len(train))]
  val_indices = [i for i in range(len(train), len(train) + len(val))]

  # Now we define samplers and loaders for train and val
  train_sampler = SubsetRandomSampler(train_indices)
  train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
  
  val_sampler = SubsetRandomSampler(val_indices)
  val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

  return train_loader, val_loader
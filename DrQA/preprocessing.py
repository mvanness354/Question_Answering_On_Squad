import spacy
import torch, torchtext
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from tqdm import tqdm
import numpy as np

import nltk
nltk.download("punkt")
from nltk.stem import LancasterStemmer

embed_model = torchtext.vocab.GloVe()
tokenizer = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "textcat"])

"""
Function to get GloVe embeddings for a list of token strings,
with option to pad with all-zero embeddings
"""
def get_embeds(tokens, embed_model, max_length=0, pad=True):
  embeds = [embed_model[token] for token in tokens]

  if pad:
    while len(embeds) < max_length:
      embeds.append(embed_model[""])

  return torch.stack(embeds)

"""
Function to find the start and end token index in the tokenized list 
corresponding to the given character start and end index from the untokenized
string
"""
def get_answer_idx(context, answer_start_index, answer_end_index):
  answer_start_token_index = answer_end_token_index = -1 
  for i, token in enumerate(context):
    if token.idx == answer_start_index:
      answer_start_token_index = i
    if token.idx + len(token) - 1 == answer_end_index:
      answer_end_token_index = i
    if answer_start_token_index > -1 and answer_end_token_index > -1:
      return answer_start_token_index, answer_end_token_index

  return answer_start_token_index, answer_end_token_index 

"""
Function to get the maximum length of a tokenizer question and context
for the whole dataset, useful for knowing how much to pad
"""
def get_max_length(data):
  max_context = max_q = 0
  for article in tqdm(data["data"], leave=False, desc="Get Max Length"):
    for paragraph in article["paragraphs"]:
      context = tokenize(paragraph["context"])
      if len(context) > max_context:
        max_context = len(context)
      
      for qa in paragraph["qas"]:
        question = tokenize(qa["question"])
        if len(question) > max_q:
          max_q = len(question)

  return max_context, max_q

"""
Function to pad a list of tokens up to some max length
"""
def pad_tokens(tokens, length):
  tokens = [token.text for token in tokens]
  return tokens + (length - len(tokens))*[""], len(tokens)

"""
Function to get exact match feature for context representation
"""
def get_overlap(context, question, pad_length):
  stemmer = LancasterStemmer()
  q_stemmed = [stemmer.stem(token.text) for token in question]
  matches = [int(stemmer.stem(token.text) in q_stemmed) for token in context]
  matches += [0]*(pad_length - len(matches))
  return matches

"""
Function to tokenize a string with spaCy tokenizer
"""
def tokenize(s):
  return [token for token in tokenizer(s)]

"""
Function to prep data to be processed by dataloader
Returns list of tuples, one tuple for each question, with the relavent 
information for the given question.  
"""
def prep_data(data, max_p, max_q):
  count = 0
  prepped_data = []
  for article in tqdm(data, leave=False, desc="Data Prep"):
    for paragraph in article["paragraphs"]:
      context_tokens = tokenize(paragraph["context"])
      context, c_len = pad_tokens(context_tokens, max_p)
      for qa in paragraph["qas"]:
        question_tokens = tokenize(qa["question"])
        question, q_len = pad_tokens(question_tokens, max_q)
        context_matches = get_overlap(context_tokens, question_tokens, max_p)
        is_answerable = not qa["is_impossible"]
        if is_answerable:
          answer_start_char = qa["answers"][0]["answer_start"]
          answer_end_char = answer_start_char + len(qa["answers"][0]["text"]) - 1
          answer_start, answer_end = get_answer_idx(context_tokens, answer_start_char, answer_end_char)
          if answer_start == answer_end == -1:
            count += 1
            continue
          else:
            # Shift because first token with be no answer token
            answer_start += 1
            answer_end += 1
        else:
          answer_start = answer_end = 0

        prepped_data.append((context, c_len, context_matches, question, q_len, is_answerable, answer_start, answer_end))

  print(count)
  return prepped_data


"""
Function to find the predicted answer given the outputted start and end
probabilities.
"""
def find_answer(start_probs, end_probs, lengths):
  results = []
  for batch in range(start_probs.shape[0]):
    min = np.inf
    min_i, min_j = -1, -1
    for i in range(1, lengths[batch]):
      for j in range(i, i+16):
        if j < lengths[batch]:
          if start_probs[batch, i] * end_probs[batch, j] < min:
            min = start_probs[batch, i] * end_probs[batch, j]
            min_i, min_j = i, j

    if min < start_probs[batch, 0] * end_probs[batch, 0]:
      results.append( (min_i, min_j) )
    else:
      results.append( (0, 0) )

  return results


class QADataset(Dataset):

  def __init__(self, data, embed_model):
    self.data = data

    self.len = len(data)
    self.stemmer = LancasterStemmer()
    self.embed_model = embed_model

  def __len__(self):
    return self.len

  def __getitem__(self, index):

    # They are already padded, so don't need to pad again
    context = get_embeds(self.data[index][0], self.embed_model, pad=False)
    question = get_embeds(self.data[index][3], self.embed_model, pad=False)

    is_answerable = self.data[index][5]
    answer_start = self.data[index][6]
    answer_end = self.data[index][7]
    c_lens = self.data[index][1]
    q_lens = self.data[index][4]

    matches = self.data[index][2]

    context = torch.stack([torch.cat( [ p, torch.tensor([match]) ] ) for p, match in zip(context, matches)])

    return context, c_lens, question, q_lens, is_answerable, answer_start, answer_end, index


def get_data_loaders(train, val, batch_size=16):
  embed_model = torchtext.vocab.GloVe()

  # First we create the dataset given our train and validation lists
  dataset = QADataset(train + val, embed_model)

  # Then, we create a list of indices for all samples in the dataset
  train_indices = [i for i in range(len(train))]
  val_indices = [i for i in range(len(train), len(train) + len(val))]

  # Now we define samplers and loaders for train and val
  train_sampler = SubsetRandomSampler(train_indices)
  train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
  
  val_sampler = SubsetRandomSampler(val_indices)
  val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

  return train_loader, val_loader

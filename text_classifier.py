from transformers import BertTokenizer, BertModel
import torch

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

example_text = 'If it quacks like a duck, it is probably a duck.'
bert_input = tokenizer(example_text, padding='max_length', max_length=20,
                       truncation=True, return_tensors="pt")

print(bert_input)
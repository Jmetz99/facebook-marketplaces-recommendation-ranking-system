import torch
from transformers import BertTokenizer
from transformers import BertModel

def process_text(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    encoded = tokenizer.__call__([text], max_length=50, padding='max_length', truncation=True)
    encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
    with torch.no_grad():
        description = model(**encoded).last_hidden_state.swapaxes(1,2)
    return description
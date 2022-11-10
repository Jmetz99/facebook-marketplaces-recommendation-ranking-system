import torch
import os
import pandas as pd
from transformers import BertTokenizer
from transformers import BertModel


class TextDatasetBert(torch.utils.data.Dataset):
    def __init__(self, 
                 root_dir: str = 'data/data_csvs/full_df.csv', 
                 labels_level: int = 0, 
                 max_length: int = 50):

        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"The file {self.root_dir} does not exist")
        products = pd.read_csv(self.root_dir, lineterminator = '\n')
        products['category'] = products['category'].apply(lambda x: self.get_category(x, labels_level))
        self.labels = products['category'].to_list()
        self.descriptions = products['product_description'].to_list()

        self.num_classes = len(set(self.labels))
        self.encoder = {y: x for x,y in enumerate(set(self.labels))}
        self.decoder = {x:y for x,y in enumerate(set(self.labels))}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.model.eval()
        self.max_length = max_length

    def __get_item__(self ,index):
        label = self.labels[index]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        sentence = self.descriptions[index]
        encoded = self.tokenizer.__call__([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        
        description = description.squeeze(0)

        return description, label

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def get_category(x, level: int = 0):
        return x.split('/')[level].strip()

if __name__ == '__main__':
    dataset = TextDatasetBert(labels_level=0)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=24,
                                             shuffle=True, num_workers=1)
    for i, (data, labels) in enumerate(dataloader):
        print(data)
        print(labels)
        print(data.size())
        if i == 0:
            break
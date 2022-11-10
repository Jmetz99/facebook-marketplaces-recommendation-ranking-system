from data_cleaning.clean_images import resize_image
from image_classifier_model import training_transforms
import os
import pandas as pd
from PIL import Image
import torch
from transformers import BertTokenizer
from transformers import BertModel
import torch.utils.data 

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root_dir: str = 'data/data_csvs/full_df.csv', 
                 labels_level: int = 0, 
                 max_length: int = 50,
                 transforms: list = training_transforms):

        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"The file {self.root_dir} does not exist")
        self.products = pd.read_csv(self.root_dir, lineterminator = '\n')
        self.products['category'] = self.products['category'].apply(lambda x: self.get_category(x, labels_level))
        self.labels = self.products['category'].to_list()
        self.descriptions = self.products['product_description'].to_list()
        self.num_classes = len(set(self.labels))
        self.encoder = {y: x for x,y in enumerate(set(self.labels))}
        self.decoder = {x:y for x,y in enumerate(set(self.labels))}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.model.eval()
        self.max_length = max_length
        self.transforms = transforms
        self.image_ids = self.products['id']

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = self.labels[index]
        label = self.encoder[label]
        label_as_tensor = torch.as_tensor(label)
        sentence = self.descriptions[index]
        encoded = self.tokenizer.__call__([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        description = description.squeeze(0)
        image_id = self.image_ids[index]
        im = Image.open(f'/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/data/training_image_data/{label}/{image_id}.jpg_resized.jpg')
        resized = resize_image(512, im)
        transformed_image = training_transforms(resized)
        image = torch.unsqueeze(transformed_image, 0)

        return (image, description), label_as_tensor

    @staticmethod
    def get_category(x, level: int = 0):
        return x.split('/')[level].strip()
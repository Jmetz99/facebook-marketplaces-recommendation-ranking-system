import torch.nn as nn
import torch.optim as optim
import torch
from multimodal_dataloader import MultiModalDataset
from torch.utils.data import DataLoader
import numpy as np
import datetime
import pickle
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CombinedModel(torch.nn.Module):
    def __init__(self,
                 num_classes: int = 13,
                 input_size: int=768):
        super(CombinedModel, self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = self.resnet50.fc.out_features
        self.image_linear = nn.Linear(out_features, 128)
        self.image_model = nn.Sequential(self.resnet50, nn.ReLU(), self.image_linear, nn.ReLU())

        self.text_model = nn.Sequential(nn.Conv1d(input_size, 256, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(256, 128, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(128, 64, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(64, 32, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(64, 128),
                                    nn.ReLU())

        self.main = nn.Sequential(nn.Linear(256, num_classes))

    def forward(self, X: tuple):
        image_inputs = self.image_model(X[0])
        text_inputs = self.text_model(X[1])
        comb_inputs = torch.cat((image_inputs, text_inputs), 1)
        return self.main(comb_inputs)


def train(model, data_loader, epochs=6):

    writer = SummaryWriter()

    optimiser = optim.Adam(model.parameters(), lr=0.003)

    criterion = nn.CrossEntropyLoss()

    batch_index = 0

    for epoch in range(epochs):
        accuracy = []
        pbar = tqdm(data_loader, total=len(data_loader))
        for ((image_features, text_features), labels) in pbar:
            image_features.to(device)
            text_features.to(device)
            labels.to(device)
            optimiser.zero_grad()
            predictions = model((image_features, text_features))
            loss = criterion(predictions, labels)
            loss.backward()
            optimiser.step()
            accuracy_batch = torch.sum(torch.argmax(predictions, dim=1) == labels).item()/len(labels)
            accuracy.append(accuracy_batch)
            ave_accuracy = np.mean(accuracy)
            writer.add_scalar('Average Accuracy', ave_accuracy, batch_index)
            pbar.set_description(str(ave_accuracy))
            batch_index += 1
        
        now = str(datetime.datetime.now().time())
        path = f'model_evaluation/multimodal_model_evaluations/multimodal_model_at_{now}_epoch:{epoch}'
        os.mkdir(path)
        torch.save(model.state_dict(), f'{path}/model_state_dict.pt')
        torch.save(optimiser.state_dict(), f'{path}/opt_date_dict.pt')
        with open(f'{path}/accuracy.txt', 'w') as f:
            f.write(f'{ave_accuracy}')

def test(model, validation_dataloader):
    writer = SummaryWriter()
    batch_index = 0
    with torch.no_grad():
        accuracy = []
        n_samples = 0
        for (image_features, text_features), labels in validation_dataloader:
                predictions = model((image_features, text_features))
                accuracy_batch = torch.sum(torch.argmax(predictions, dim=1) == labels).item()/len(labels)
                accuracy.append(accuracy_batch)
                ave_accuracy = np.mean(accuracy)
                writer.add_scalar('Average Test Accuracy', ave_accuracy, batch_index)
                print(ave_accuracy)
                batch_index += 1

def split_train_test(dataset, train_percentage: float = 0.8):
    train_split = int(len(dataset) * train_percentage)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_split, len(dataset) - train_split]
)
    return train_dataset, validation_dataset


if __name__ == '__main__':
    dataset = MultiModalDataset()
    train_dataset, validation_dataset = split_train_test(dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=24, shuffle=True)

    model = CombinedModel()
    model.load_state_dict(torch.load('/Users/jacobmetz/Documents/GitHub/facebook-marketplaces-recommendation-ranking-system/model_evaluation/multimodal_model_evaluations/multimodal_model_at_01:48:41.515355_epoch:4/weights.pt'))
    model.eval()
    train(model, train_dataloader)
    test(model, validation_dataloader)

    with open('multi_decoder.pkl', 'wb') as f:
        pickle.dump(dataset.decoder, f)
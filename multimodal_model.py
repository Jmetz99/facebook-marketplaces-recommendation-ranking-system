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
        self.image_linear = nn.Linear(out_features, 32)
        self.image_model = nn.Sequential(self.resnet50, self.image_linear)

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
                                    nn.Linear(64, 32),
                                    nn.ReLU())

        self.linear = nn.Linear(64, num_classes)

    def forward(self, X: tuple):
        image_inputs = self.image_model(X[0])
        text_inputs = self.text_model(X[1])
        comb_inputs = torch.cat((image_inputs, text_inputs), 1)
        return self.linear(comb_inputs)


def train(model, data_loader, epochs=5):
    writer = SummaryWriter()

    optimiser = optim.Adam(model.parameters(), lr=0.001)

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
        torch.save(model.state_dict(), f'{path}/weights.pt')
        with open(f'{path}/accuracy.txt', 'w') as f:
            f.write(f'{ave_accuracy}')

if __name__ == '__main__':
    dataset = MultiModalDataset()
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
    model = CombinedModel()
    train(model, dataloader)
    with open('multi_decoder.pkl', 'wb') as f:
        pickle.dump(dataset.decoder, f)
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

Data = np.random.randint((4, 2))
Label = np.array([[0], [1], [2], [3]])

class subDataset(Dataset):

    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.IntTensor(self.Label[index])
        return data, label


dataset = subDataset(Data, Label)
print(dataset)
print("dataset's length：", dataset.__len__())
print(dataset.__getitem__(0))
print(dataset[0])

dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)
for i, item in enumerate(dataloader):
    print('i:', i)
    data, label = item
    print('data:', data)
    print('label:', label)
from torch.utils.data import Dataset

class BatchData(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = self.data_list[item]
        input = data['input']
        label = data['label']

        data_info = {'input': input, 'label': label}
        return data_info

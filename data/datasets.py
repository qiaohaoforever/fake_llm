from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = self.tokenizer.encode_plus(
            item['Content'],
            max_length=self.max_length,
            padding = 'max_length',
            truncation=True,
            return_tensors='pt'
        )
        label = torch.tensor(int(item['Label']), dtype=torch.long)
        return {
            'input_ids': context['input_ids'].squeeze(),
            'attention_mask': context['attention_mask'].squeeze(),
            'labels': label
        }


def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [eval(line) for line in f]
    return data


def get_dataloaders(file_path, tokenizer, batch_size=64, test_size=0.2):
    data = load_data(file_path)
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=100)

    train_dataset = MyDataset(train_data, tokenizer)
    val_dataset = MyDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/qiao/.cache/modelscope/hub/tiansz/bert-base-chinese/', local_files_only=True)

    file_path = 'raw/train_data_32k.txt'
    train_loader, val_loader = get_dataloaders(file_path, tokenizer)
    for batch in train_loader:
        print(batch)
        break

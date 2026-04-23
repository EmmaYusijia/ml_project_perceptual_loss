import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset


# purpose: convert CIFAR-10 dataset samples into (grayscale_image, color_image) pairs
# output: dataset where each item is (grayscale_image, color_image)
# side effect: None
class ColorizationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.grayscale_transform = transforms.Grayscale(num_output_channels=1)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        color_image, _ = self.dataset[index]
        grayscale_image = self.grayscale_transform(color_image)
        return grayscale_image, color_image


# purpose: load CIFAR-10, split, wrap and return dataloaders for training, validation, and testing
# output:
#   training_loader: DataLoader for training set
#   validation_loader: DataLoader for validation set
#   testing_loader: DataLoader for test set
# side effect: downloads CIFAR-10 dataset if not exist
def build_dataloaders(batch_size=64,root="./data",training_ratio=0.9,seed=42):
    all_train_data = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())
    test_data = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())
    train_count = int(training_ratio * len(all_train_data))
    train_subset, validation_subset = random_split(
        all_train_data,[train_count, len(all_train_data) - train_count],generator=torch.Generator().manual_seed(seed))
    train_subset = Subset(train_subset, list(range(5000)))
    validation_subset = Subset(validation_subset, list(range(1000)))
    test_data = Subset(test_data, list(range(1000)))
    training_loader = DataLoader(ColorizationDataset(train_subset), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(ColorizationDataset(validation_subset), batch_size=batch_size, shuffle=False)
    testing_loader = DataLoader(ColorizationDataset(test_data), batch_size=batch_size, shuffle=False)
    return training_loader, validation_loader, testing_loader


# purpose: test the data pipeline by loading and printing batch shapes
# output: None
# side effect: prints batch shapes
if __name__ == "__main__":
    training_loader, validation_loader, testing_loader = build_dataloaders(batch_size=8)
    grayscale_batch, color_batch = next(iter(training_loader))
    print("grayscale batch shape:", grayscale_batch.shape)
    print("color batch shape:", color_batch.shape)

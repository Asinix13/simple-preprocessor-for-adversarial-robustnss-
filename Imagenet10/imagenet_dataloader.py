import torch
from torchvision import datasets, transforms


def get_imagenet10_loader(root, data_dim, train_test_split, batch_size):


    transform = transforms.Compose([
    transforms.Resize(256),           
    transforms.RandomResizedCrop(224), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])


    dataset = datasets.ImageFolder(root=root, transform=transform)
    print(len(dataset))
    # print(dataset.imgs[0])

    generator = torch.Generator().manual_seed(42)
    train_set, test_set = torch.utils.data.random_split(
        dataset,
        train_test_split, 
        generator= generator) 
    print(len(train_set))
    print(len(test_set))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False)
    
    return train_loader, test_loader
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import ssl
from torchvision.transforms import functional as F
from torchvision.transforms.functional import to_pil_image
import numpy as np




def combine_images(data, combination_type, label_type = 'sum'):
    combined_images = []
    combined_labels = []
    
    if combination_type == 'random':
        indices = list(range(len(data)))
        random.shuffle(indices)
        for i in range(0, len(indices) - 1, 2):
            img1, label1 = data[indices[i]]
            img2, label2 = data[indices[i+1]]
            
            combined_img = torch.cat((img1, img2), dim=1)
            combined_img_flipped = torch.cat((img2, img1), dim=1)
            
            combined_lab = label1 + label2
            
            combined_images.append(combined_img)
            combined_images.append(combined_img_flipped)
            combined_labels.append(combined_lab)
            combined_labels.append(combined_lab)
            
    return combined_images, combined_labels


def show_digit(image_no, images_dataset, label_dataset):
    image_np = images_dataset[image_no].numpy().squeeze()
    plt.imshow(image_np, cmap='gray')
    plt.title(f'Label {label_dataset[image_no]}')
    plt.axis('off')
    plt.show()



def split_dataset(image_dataset, label_dataset, train_ratio = 0.6, val_ratio=0.2):
    train_no = int(len(image_dataset)*train_ratio)
    val_no = train_no + int(len(image_dataset)*val_ratio)
    
    train_images = image_dataset[:train_no]
    train_labels = label_dataset[:train_no]
    
    val_images = image_dataset[train_no:val_no]
    val_labels = label_dataset[train_no:val_no]
    
    test_images = image_dataset[val_no:]
    test_labels = label_dataset[val_no:]
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
    

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        noise = torch.randn(image.size()) * self.std + self.mean
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0.0, 1.0)
    
class AddSaltAndPepperNoise:
    def __init__(self, prob=0.05):
        self.prob = prob

    def __call__(self, image):
        noisy_image = image.clone()
        num_pixels = image.numel()

        # Add salt noise (white pixels)
        num_salt = int(self.prob * num_pixels / 2)
        salt_coords = [torch.randint(0, dim, (num_salt,)) for dim in image.shape]
        noisy_image[salt_coords] = 1.0

        # Add pepper noise (black pixels)
        num_pepper = int(self.prob * num_pixels / 2)
        pepper_coords = [torch.randint(0, dim, (num_pepper,)) for dim in image.shape]
        noisy_image[pepper_coords] = 0.0

        return noisy_image




def show_images_side_by_side(index, combined_dataset, train_images, train_labels):
    """
    Display two images side by side: one from the combined dataset and one from the original dataset.

    Args:
        index (int): Index of the image to display.
        combined_dataset (list): List of combined images (e.g., augmented and original).
        train_images (list): List of original images.
        train_labels (list): List of labels.
    """
    # Create a figure
    plt.figure(figsize=(8, 4))

    # Plot the combined dataset image
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    plt.imshow(combined_dataset[index].squeeze(), cmap='gray')
    plt.title(f"Original (Label: {train_labels[index]})")
    plt.axis('off')

    # Plot the original dataset image
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    plt.imshow(combined_dataset[index+42000].squeeze(), cmap='gray')
    plt.title(f"Augmented (Label: {train_labels[index]})")
    plt.axis('off')

    # Show the plots
    plt.show()


def prepare_data(train_images, train_labels, val_images, val_labels, batch_size):
    train_data_stacked = torch.stack(train_images)
    train_labels_stacked = torch.tensor(train_labels)

    val_data_stacked = torch.stack(val_images)
    val_labels_stacked = torch.tensor(val_labels)

    train_tensor = TensorDataset(train_data_stacked, train_labels_stacked)
    val_tensor = TensorDataset(val_data_stacked, val_labels_stacked)

    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader



def download_dataset():

    ssl._create_default_https_context = ssl._create_unverified_context

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    return full_dataset



def augment_dataset(train_images):
    augmented_transform = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
    AddGaussianNoise(mean=0.0, std=0.05),
    AddSaltAndPepperNoise(prob=0.05),
    transforms.Normalize((0.5,), (0.5,))
    ])

    augmented_images = []

    for tensor_image in train_images:
        pil_image = F.to_pil_image(tensor_image)
        augmented_image = augmented_transform(pil_image)  # Apply transform pipeline
        augmented_images.append(augmented_image)

    return augmented_images




def prepare_data_to_numpy(train_images, train_labels, val_images, val_labels, test_images, test_labels):
    train_X = torch.stack(train_images).view(len(train_images), -1).numpy()
    train_y = np.array(train_labels)

    val_X = torch.stack(val_images).view(len(val_images), -1).numpy()
    val_y = np.array(val_labels)

    test_X = torch.stack(test_images).view(len(test_images), -1).numpy()
    test_y = np.array(test_labels)
    
    return train_X, train_y, val_X, val_y, test_X, test_y

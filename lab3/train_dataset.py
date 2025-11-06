from torchvision.datasets.mnist import FashionMNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from dino_augmentation import DinoAugmentation
import torch

dataset_mean = (0.5,)
dataset_std = (0.5,)

def preview_augmented_fashionmnist(views):
    n_views = len(views)
    # Plot
    fig, axes = plt.subplots(1, n_views, figsize=(2.5 * n_views, 3))
    for i in range(n_views):
        ax = axes[i]
        view = views[i]
        # Convert tensor to numpy for plotting
        if isinstance(view, torch.Tensor):
            view = view.squeeze().detach().cpu().numpy()
        ax.imshow(view, cmap="gray")
        ax.axis("off")
        ax.set_title(f"View {i+1}")
    # plt.suptitle(f'FashionMNIST label: {label}', fontsize=14)
    plt.show()


base_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=1)], p=0.25),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)],
            p=0.25,
        ),
        transforms.RandomResizedCrop(size=28, scale=(0.7, 1), ratio=(1, 1)),
    ]
)

student_transforms = transforms.Compose(
    [
        transforms.RandomPerspective(0.3, p=0.5),
        transforms.RandomResizedCrop(size=28, scale=(0.1, 0.4), ratio=(1, 1)),
    ]
)

end_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(dataset_mean, dataset_std)]
)

n_global_views = 2
n_cropped_views = 5

dino_transforms = DinoAugmentation(
    base_transforms=base_transforms,
    student_augs=student_transforms,
    final_transforms=end_transforms,
    n_global_views=n_global_views,
    n_student_crops=n_cropped_views,
)

train_dataset = FashionMNIST(
    root="./data", download=True, train=True, transform=dino_transforms
)

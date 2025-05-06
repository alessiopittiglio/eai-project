import torchvision.transforms as T

def build_transforms(config: dict, augment: bool) -> T.Compose:
    img_size = config.get('img_size', 224)
    norm_mean = config.get('norm_mean', (0.485, 0.456, 0.406))
    norm_std = config.get('norm_std', (0.229, 0.224, 0.225))
    should_augment = augment and config.get('train_augmentations', True)

    normalize = T.Normalize(mean=norm_mean, std=norm_std)
    transforms_list = [T.Resize((img_size, img_size))]

    if should_augment:
        # Add here the desired augmentations for training
        if config.get('aug_use_horizontal_flip', True):
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))

    transforms_list.extend([
        T.ToTensor(), # Converts PIL [0,255] HWC to Tensor [0,1] CHW
        normalize
    ])
    return T.Compose(transforms_list)

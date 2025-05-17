import torchvision.transforms as T

def build_transforms(config: dict, augment: bool) -> T.Compose:
    """
    Args:
        config (dict): Dictionary containing the parameters for the
            transformations
        augment (bool): Whether to apply augmentations or not

    Returns:
        T.Compose: Composed transformation
    """
    
    # Retrieve params
    config_img_size = config.get('img_size', 224)
    img_size = (config_img_size, config_img_size) if isinstance(config_img_size, int) else config_img_size
    norm_mean = config.get('norm_mean', (0.485, 0.456, 0.406))
    norm_std = config.get('norm_std', (0.229, 0.224, 0.225))
    should_augment = augment and config.get('train_augmentations', True)
    
    # Define the transformations
    transforms_list = [
        T.Resize(img_size)
        ]
    
    # Add augmentations if in training mode
    if should_augment:
        # Add here the desired augmentations for training
        if config.get('aug_use_horizontal_flip', True):
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))

    # Transform to Tensor
    # Note: The order of transformations is important.
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
        #T.Lambda(lambda x: x.permute(2, 0, 1)),  # Change from (H, W, C) to (C, H, W)
        #T.Lambda(lambda x: x.unsqueeze(0))  # Add a channel dimension for 3D models
    ])
    
    # Create the composed transform
    return T.Compose(transforms_list)

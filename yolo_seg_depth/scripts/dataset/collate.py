import torch

def collate_fn(batch):
    """
    Collates data samples into batches.
    
    Note:
    Output dimension (along 0th axis) of `stacked_images`
    may or may not be equal to the `concatenated_targets`.
    
    """
    images, targets = zip(*batch)

    # Stack images along the new dimension 0; shape = (B, C, H, W)
    stacked_images = torch.stack(images, dim=0)
    
    # Concatenate the tensors along newly created dimension 0
    # which represents the batch index; shape = (Num_batch_Boxes, 6)
    # [batch_id, class_index, xn, yn, wn, hn]
    concatenated_targets = torch.cat([
        torch.cat([idx * torch.ones(target.size(0), 1), target], dim=1)
        for idx, target in enumerate(targets)
    ], dim=0)

    return stacked_images, concatenated_targets
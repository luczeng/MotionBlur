def collate_fn_varying_size(batch):

    images = [item["image"] for item in batch]
    gt = [item["gt"] for item in batch]

    return images, gt

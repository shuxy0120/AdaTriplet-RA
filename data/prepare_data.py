import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.folder_new import ImageFolder_new
from data.imagelist import  ImageList
from torch.utils.data import DataLoader
import data.pre_process as prep


def generate_dataloader(args):
    # Data loading code

    domainnet = 0

    if domainnet == 1:
        dsets = {}
        dset_loaders = {}
        prep_dict = {}
        config = {}
        config["data"] = {
            "source": {"list_path": '../symnets/dataset/Domainnet/splits_mini2/painting_train_mini.txt', "batch_size": 64}, \
            "target": {"list_path": '../symnets/dataset/Domainnet/splits_mini2/real_train_mini.txt', "batch_size": 64}, \
            "source_test": {"list_path": '../symnets/dataset/Domainnet/splits_mini2/painting_test_mini.txt', "batch_size": 64}, \
            "test": {"list_path": '../symnets/dataset/Domainnet/splits_mini2/real_test_mini.txt', "batch_size": 64}}
        config["prep"] = {'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}

        data_config = config["data"]
        prep_config = config["prep"]
        if "webcam" in data_config["source"]["list_path"] or "dslr" in data_config["source"]["list_path"]:
            prep_dict["source"] = prep.image_train(**config["prep"]['params'])
        else:
            prep_dict["source"] = prep.image_target(**config["prep"]['params'])

        if "webcam" in data_config["target"]["list_path"] or "dslr" in data_config["target"]["list_path"]:
            prep_dict["target"] = prep.image_train(**config["prep"]['params'])
        else:
            prep_dict["target"] = prep.image_target(**config["prep"]['params'])
        dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                    transform=prep_dict["source"])
        dset_loaders["source"] = DataLoader(dsets["source"], batch_size=args.batch_size_s, \
                                            shuffle=True, num_workers=args.workers, drop_last=True)

        dsets["source_test"] = ImageList(open(data_config["source_test"]["list_path"]).readlines(), \
                                         transform=prep_dict["source"])
        dset_loaders["source_test"] = DataLoader(dsets["source_test"], batch_size=args.batch_size_t, \
                                                 shuffle=False, num_workers=args.workers, drop_last=True)

        dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                    transform=prep_dict["target"])
        dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size_t, \
                                            shuffle=True, num_workers=args.workers, drop_last=True)

        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(),
                                  transform=prep_dict["target"])
        dset_loaders["test"] = DataLoader(dsets["target"], batch_size=args.batch_size_t, \
                                          shuffle=False, num_workers=args.workers)

        return dset_loaders["source"], dset_loaders["source_test"], dset_loaders["target"], dset_loaders["test"]

    traindir_source = os.path.join(args.data_path_source, args.src)
    traindir_target = os.path.join(args.data_path_source_t, args.src_t)
    valdir = os.path.join(args.data_path_target, args.tar)
    if not os.path.isdir(traindir_source):
        # split_train_test_images(args.data_path)
        raise ValueError('Null path of source train data!!!')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    source_train_dataset = datasets.ImageFolder(
        traindir_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=args.batch_size_s, shuffle=True,
        drop_last=True, num_workers=args.workers, pin_memory=True, sampler=None
    )

    source_val_dataset = ImageFolder_new(
        traindir_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    source_val_loader = torch.utils.data.DataLoader(
        source_val_dataset, batch_size=args.batch_size_s, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None
    )
    
    target_train_dataset = datasets.ImageFolder(
        traindir_target,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset, batch_size=args.batch_size_t, shuffle=True,
        drop_last=True, num_workers=args.workers, pin_memory=True, sampler=None
    )
    target_val_loader = torch.utils.data.DataLoader(
        ImageFolder_new(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size_t, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    return source_train_loader, source_val_loader, target_train_loader, target_val_loader


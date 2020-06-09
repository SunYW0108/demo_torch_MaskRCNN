import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import os.path as osp
from engine import train_one_epoch, evaluate
import utils, ECP_Instance_Dataset, transforms as T
import subprocess
import shlex
import yaml
import argparse
import datetime
import tqdm

def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

here = osp.dirname(osp.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--resume', help='checkpoint path')
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument(
        '--num_epochs', type=int, default=1000, help='num_epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=0.005, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='momentum',
    )
    args = parser.parse_args()

    args.model = 'ECP_MaskRCNN'
    args.git_hash = git_hash()

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.manual_seed(1337)
    if (device.type=='cuda'):
        torch.cuda.manual_seed(1337)

    # 1. dataset
    # our dataset has two classes only - background and windows
    num_classes = 2
    # use our dataset and defined transformations
    root='/home/sun/facade_datasets/2.ECP'
    dataset = ECP_Instance_Dataset.ECP_Instance_Dataset(root, get_transform(train=True))
    dataset_test = ECP_Instance_Dataset.ECP_Instance_Dataset(root, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-30])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-30:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                              shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    # 2. model
    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']

    model.to(device)

    # 3. optimizer
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)


    # 4. train
    interval_validate=10

    for epoch in tqdm.trange(start_epoch, args.num_epochs,
                                 desc='Train', ncols=80):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

        if epoch % interval_validate == 0:
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)

    print("That's it!")

if __name__ == '__main__':
    main()
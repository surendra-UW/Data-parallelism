import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import model as mdl
import time

device = "cpu"
torch.set_num_threads(4)

batch_size = 64 # batch for one node

def initialize_gloo_backend(args):
    world_size = args.num_nodes
    rank = args.rank
    master_ip = args.master_ip
    port = 7500
    init_method = f'tcp://{master_ip}:{port}'
    dist.init_process_group(backend=dist.Backend.GLOO, init_method = init_method, rank=rank, world_size=world_size)


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # Your code goes here!
        #starting a timer to measure time
        if batch_idx == 1:
            start_time = time.time()

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()

        #sync gradients using allreduce()
        for param in model.parameters():
            gatheredTensorList = [torch.zeros_like(param.grad.data) for _ in range(dist.get_world_size())]
             
            if dist.get_rank() == 0:
                dist.gather(param.grad.data, gather_list=gatheredTensorList, dst=0)
                summed_grad = torch.sum(torch.stack(gatheredTensorList, dim=0), dim=0)
                final_grad = summed_grad/dist.get_world_size()
                scatterList = [final_grad for _ in range(dist.get_world_size())]
                dist.scatter(param.grad.data, scatter_list=scatterList, src=0)
            else:
                dist.gather(param.grad.data, dst=0)
                dist.scatter(param.grad.data ,src=0)

        optimizer.step()

        if batch_idx == 39:
            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Loss at iteration {batch_idx+1} is: {loss.item()}\nAverage time per iteration: {time_taken/39}s")
            break
        elif (batch_idx+1) % 10 == 0:
            print(f"Loss at iteration {batch_idx+1} is: {loss.item()}")

    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main():
    torch.manual_seed(0)
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    
    sampler = DistributedSampler(training_set)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              sampler=sampler,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    torch.manual_seed(0)
    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--master-ip')
    parser.add_argument('--num-nodes', type=int)
    parser.add_argument('--rank', type=int)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #parse the arguments
    args = parse_arguments()

    #initialize the gloo backend
    initialize_gloo_backend(args)

    #run the model
    main()

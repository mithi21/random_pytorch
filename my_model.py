from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Import custom layers
from convolution2d import CustomConv2d
from avgpool2d import CustomAvgPool2d
from dropout import CustomDropout
from relu import CustomReLU
from maxpool2d import CustomMaxPool2d

class Net1(nn.Module):
    def __init__(self, drop=0.01):
        super(Net1, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # You can replace this with CustomAvgPool2d if needed
        
        self.convblock1 = nn.Sequential(
            CustomConv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False),#[B,C,H,W]
            nn.BatchNorm2d(8),#[B,C,H,W]
            CustomReLU(),#[B,C,H,W]
            CustomDropout(drop)#[B,C,H,W]
        ) 

        self.convblock2 = nn.Sequential(
            CustomConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, bias=False),  
            nn.BatchNorm2d(16),
            CustomReLU(),
            CustomDropout(drop)
        ) 

        self.trans1 = nn.Sequential(
            CustomConv2d(in_channels=16, out_channels=8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            CustomReLU(),
            CustomDropout(drop)
        )

        self.convblock3 = nn.Sequential(
            CustomConv2d(in_channels=8, out_channels=8, kernel_size=3, padding=0, bias=False), # output_size = 10    RF:  9
            nn.BatchNorm2d(8),
            CustomReLU(),
            CustomDropout(drop)
        ) 

        self.convblock4 = nn.Sequential(
            CustomConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0, bias=False), # output_size = 10    RF:  13
            nn.BatchNorm2d(16),
            CustomReLU(),
            CustomDropout(drop)
        )

        self.convblock5 = nn.Sequential(
            CustomConv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0, bias=False), # output_size = 10    RF:  17
            nn.BatchNorm2d(16),
            CustomReLU(),
            CustomDropout(drop)
        )

        self.gap = CustomAvgPool2d(kernel_size=8)  # Replacing AvgPool2d with CustomAvgPool2d
        
        self.convblock6 = nn.Sequential(
            CustomConv2d(in_channels=16, out_channels=10, kernel_size=1, bias=False),
        )

    def forward(self, x):
        """
        x = self.convblock1(x) #3
        x = self.convblock2(x) #5
        x = self.pool(x) #6
        x = self.trans1(x)
        x = self.convblock3(x) # 10
        x = self.convblock4(x) # 14
        x = self.convblock5(x) # 18
        print(x.shape)
        print("gap")
        x = self.gap(x)
        print(x.shape)
        x = self.convblock6(x) 
        x = x.view(-1, 10)   # convert 2D to 1D
        
        return F.log_softmax(x, dim=-1)
        
        """
        x = self.convblock1(x) #3
        x = self.convblock2(x) #5
        x = self.pool(x) #6
        x = self.trans1(x)
        x = self.convblock3(x) # 10
        x = self.convblock4(x) # 14
        x = self.convblock5(x) # 18
        print(x.shape)
        print("gap")
        x = self.gap(x)
        print(x.shape)
        x = self.convblock6(x) 
        x = x.view(-1, 10)   # convert 2D to 1D
        
        return F.log_softmax(x, dim=-1)
    

def total_params(model):
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
    return total_params


from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))


if __name__ == "__main__":
    model = Net1()
    input = torch.randn(1, 1, 28, 28)
    output = model(input)
    print(output.shape)
    print(f" total params {total_params(model)}")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    model = Net1().to(device)

    print(f" total params {total_params(model)}")
    from __future__ import print_function
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    # Train Phase transformations
    train_transforms = transforms.Compose([
                                            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
                                        ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
                                        #  transforms.Resize((28, 28)),
                                        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
    train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    SEED = 1

    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=96, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    
    # from torchsummary import summary
    # summary(model, (1, 28, 28))

    from torch.optim.lr_scheduler import StepLR

    model =  Net1().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9)
    # # optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    # scheduler = StepLR(optimizer, step_size=4, gamma=0.1)
    EPOCHS = 1
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) 

    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train(model, device, train_loader, optimizer, epoch)
        # scheduler.step()
        test(model, device, test_loader)
    
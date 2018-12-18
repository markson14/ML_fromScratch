import torch as torch
import torchvision

# HyperParameters
epoches = 10
batch_size = 16
LR = 0.001

# Dataset
train_data = torchvision.datasets.MNIST(
    root='./cifar10',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = True)

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28*28,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,12),
            torch.nn.ReLU(),
            torch.nn.Linear(12,3),
            torch.nn.ReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3,12),
            torch.nn.ReLU(),
            torch.nn.Linear(12,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,28*28),
            torch.nn.Sigmoid(),
        )
    
    def forward(self,x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode,decode
    
vae = AutoEncoder()
optimizer = torch.optim.Adam(vae.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()

for epoch in range(epoches):
    for step, (x,b_label) in enumerate(train_data):
        b_x = x.view(-1, 28*28)
        b_y = x.view(-1, 28*28)
        print(b_x, b_y)
        encode, decode = vae(b_x)

        loss = loss_func(decode, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print("Epoch",epoch, '| train loss: %0.4f' % loss.data[0])




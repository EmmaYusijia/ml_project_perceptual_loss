from data import build_dataloaders
import torch
from torch import nn
from torch.optim import Adam
from training import *
from perceptual_loss import *

# Mini-Batch SGD hyperparameters
lr = 0.001

def double_conv(input_channel, output_channel):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, 3, padding="same"), 
        nn.ReLU(),  
        nn.Conv2d(output_channel, output_channel, 3, padding="same"), 
        nn.ReLU()    
    )

def up_trans(input_channel, output_channel):
    return nn.Sequential(
        nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2), 
        nn.ReLU()     
    )

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # contracting path
        self.max_pool = nn.MaxPool2d(2, 2)  
        self.down_conv_1 = double_conv(1,64)
        self.down_conv_2 = double_conv(64,128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256,512)
        self.down_conv_5 = double_conv(512,1024)

        # expansive path
        self.up_trans_1 = up_trans(1024, 512)
        self.up_conv_1 = double_conv(1024,512)
        self.up_trans_2 = up_trans(512, 256)
        self.up_conv_2 = double_conv(512,256)
        self.up_trans_3 = up_trans(256,128)
        self.up_conv_3 = double_conv(256,128)
        self.up_trans_4 = up_trans(128,64)
        self.up_conv_4 = double_conv(128,64)

        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, image):
        # contracting path
        x1 = self.down_conv_1(image)
        x2 = self.max_pool(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool(x7)
        x9 = self.down_conv_5(x8)

        # expansive path
        x = self.up_trans_1(x9)
        x = self.up_conv_1(torch.cat([x, x7], 1))

        x = self.up_trans_2(x)
        x = self.up_conv_2(torch.cat([x, x5], 1))
        
        x = self.up_trans_3(x)
        x = self.up_conv_3(torch.cat([x, x3], 1))

        x = self.up_trans_4(x)
        x = self.up_conv_4(torch.cat([x, x1], 1))

        x = self.out(x)

        return x

if __name__ == '__main__':
    training_loader, validation_loader, test_loader = build_dataloaders(batch_size=32)
    model = UNet()
    loss_func = perceptual_loss
    run_model(model, training_loader, validation_loader, optimizer=Adam, learning_rate=lr, get_output_and_loss=loss_func)
    visualize(model, "UNet", test_loader)
    


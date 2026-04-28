import torch
import torchvision.models as models
import torch.nn.functional as F

def perceptual_loss(model, X, Y, nn):
    train_output = model(X)

    feat_list = []
    for i in range(0, 10): # loop over the layers in the features block of vgg19
        layer = nn.features[i]

        train_output = layer(train_output) # shape: N*C*H*W
        Y = layer(Y) # shape: N*C*H*W

        if (i == 2 or i == 6):
            scaled_pred = scale(train_output)
            scaled_target = scale(Y)
            loss = ((scaled_target-scaled_pred) ** 2).sum(dim=1).mean()
            feat_list.append(loss)

    train_loss = sum(feat_list) / len(feat_list) + 0.07 * ((train_output - Y) ** 2).mean()
    
    return train_output, train_loss

def perceptual_loss_simp(model, X, Y, nn):
    train_output = model(X)

    feat_list = []
    for i in range(0, 10): # loop over the layers in the features block of vgg19
        layer = nn.features[i]

        train_output = layer(train_output) # shape: N*C*H*W
        Y = layer(Y) # shape: N*C*H*W

        if (i == 9):
            scaled_pred = scale(train_output)
            scaled_target = scale(Y)
            loss = ((scaled_target-scaled_pred) ** 2).sum(dim=1).mean()
            feat_list.append(loss)

    train_loss = sum(feat_list) / len(feat_list) + 0.06 * ((train_output - Y) ** 2).mean()
    
    return train_output, train_loss
            
def scale(tensor_obj):
    norm = torch.sqrt((tensor_obj ** 2).sum(dim=1, keepdim=True))
    scaled_obj = tensor_obj / norm

    return scaled_obj
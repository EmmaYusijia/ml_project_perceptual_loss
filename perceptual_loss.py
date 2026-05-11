import torch
import torch.nn.functional as F

def perceptual_loss(model, X, Y, network):
    train_output = model(X)

    feat_list = []
    for i in range(0, 13): # loop over the layers in the features block of alexnet
        layer = network.features[i]

        train_output = layer(train_output) # shape: N*C*H*W
        Y = layer(Y) # shape: N*C*H*W

        if (i == 1 or i == 4 or i == 7 or i == 9 or i == 11):
        # if (i == 0 or i == 3 or i == 6 or i == 8 or i == 10):
            scaled_pred = scale(train_output)
            scaled_target = scale(Y)
            loss = ((scaled_target-scaled_pred) ** 2).sum(dim=1).mean()
            feat_list.append(loss)

    train_loss = sum(feat_list) / len(feat_list)

    # print(f"perceptual loss: {feat_list}")
    
    return train_output, train_loss

def perceptual_loss_simp(model, X, Y, network):
    train_output = model(X)

    feat_list = []
    for i in range(0, 13): # loop over the layers in the features block of vgg19
        layer = network.features[i]

        train_output = layer(train_output) # shape: N*C*H*W
        Y = layer(Y) # shape: N*C*H*W

        if (i == 11):
            scaled_pred = scale(train_output)
            scaled_target = scale(Y)
            loss = ((scaled_target-scaled_pred) ** 2).sum(dim=1).mean()
            feat_list.append(loss)

    train_loss = sum(feat_list) / len(feat_list)
    
    return train_output, train_loss
            
def scale(tensor_obj):
    norm = torch.sqrt((tensor_obj ** 2).sum(dim=1, keepdim=True))
    scaled_obj = tensor_obj / norm

    return scaled_obj







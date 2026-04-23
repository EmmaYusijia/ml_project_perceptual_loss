import torch
import torchvision.models as models
import torch.nn.functional as F

def perceptual_loss(model, X, Y):
    train_output = model(X)
    train_output = F.interpolate(train_output, size=(224, 224), mode='nearest', 
                                 align_corners=None, recompute_scale_factor=None, antialias=False)
    target_output = F.interpolate(Y, size=(224, 224), mode='nearest', 
                                 align_corners=None, recompute_scale_factor=None, antialias=False)

    alexnet = models.alexnet(weights='DEFAULT')

    layer0 = alexnet.features[0]
    feat_pred = layer0(train_output) 
    feat_target = layer0(target_output) 

    feat_list = []
    for i in range(1, 13): # loop over the layers in the features block of AlexNet
        feature_extractor = alexnet.features[i]

        feat_pred = feature_extractor(feat_pred) # shape: N*C*H*W
        feat_target = feature_extractor(feat_target) # shape: N*C*H*W

        if (i == 7 or i == 9):
            sample_list = []
            for j in range(feat_target.shape[0]):
                x_target = feat_target[j] # shape: C*H*W
                x_pred = feat_pred[j] # shape: C*H*W
                scaled_pred, scaled_target = scale(x_pred, x_target) # shape: C*H*W
                loss = torch.sub(scaled_target, scaled_pred)
                loss = torch.sum(loss, dim=0)
                sample_list.append(loss)
            feat_loss = sum(sample_list) / len(sample_list)
            feat_list.append(feat_loss)

    train_loss = sum(feat_list) / len(feat_list) 
    
    return train_output, train_loss
            

def scale(pred, target):
    for i in range(target.shape[1]):
        for j in range(target.shape[2]):
            pred_sum = 0
            target_sum = 0
            for k in range(target.shape[0]):
                pred_sum += pred[k,i,j]**2
                target_sum += target[k,i,j]**2
            pred_scale = pred_sum ** 0.5
            target_scale = target_sum ** 0.5
            
            pred_scaled = pred[:, i, j] / pred_scale
            target_scaled = target[:, i, j] / target_scale

    return pred_scaled, target_scaled







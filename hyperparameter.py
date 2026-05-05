from data import build_dataloaders
from UNet import UNet
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam

num_epochs = 10
learning_rate = 0.001
batch_size = 32

# purpose: load the CIFAR-10 vgg19 model
# output: vgg19 model
# side effect: download the model
def load_cifar_vgg19():
    # load the vgg19 model from the repo
    train_data = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg19_bn", pretrained=True)
    # put the model in eval as we do not train it
    train_data.eval()
    # remove the parameter in the features
    for param in train_data.parameters():
        param.requires_grad = False
    return train_data

# purpose: normalize different features based on the channel dimension
# output: normalized feature map
def scale_features(features):
    # compute L2 norm of all channels and divide the features by its norm
    norm = torch.sqrt((features**2).sum(dim=1, keepdim=True) + 1e-8)
    feature_map = features/norm
    return feature_map

# purpose: create perceptual loss and MSE
# output: loss function of the training loop
def make_perceptual_loss(select_layer, max_layer, mse_weight):
    # define the loss function for training
    def loss_function(model, X, Y, train_data):
        # generate color prediction from grayscale input
        prediction = model(X)
        # generate prediction and target
        predict_feature = prediction
        target_feature = Y
        # store perceptual loss of one choice
        feature_loss_lst = []
        # add prediction and target into vgg model
        for layer_index in range(max_layer):
            # select one vgg layer choice
            layer = train_data.features[layer_index]
            # compute prediction and target feature of the layer choice
            predict_feature = layer(predict_feature)
            target_feature = layer(target_feature)
            # compute feature distance of the layer choice
            if layer_index in select_layer:
                # normalize prediction and target feature
                new_predict = scale_features(predict_feature)
                new_target = scale_features(target_feature)
                # compute feature square distance and loss
                layer_loss = ((new_target - new_predict)**2).sum(dim=1).mean()
                feature_loss_lst.append(layer_loss)
        # average loss of the layer choice
        perceptual_feature_loss = sum(feature_loss_lst)/len(feature_loss_lst)
        # compute MSE between prediction and target
        mse_loss = nn.MSELoss()(prediction, Y)
        # combine perceptual feature loss and MSE
        total_loss = perceptual_feature_loss + mse_weight * mse_loss
        return prediction, total_loss
    return loss_function

# purpose: train one model of one perceptual loss choice
# output: model, training curve, validation curve
# side effect: update model parameter
def train_one_model(model, train_loader, validation_loader, train_data, loss_function):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # store mean train loss and validation loss for each epoch
    train_epoch_loss = []
    valid_epoch_loss = []
    # loop over train epochs
    for epoch in range(num_epochs):
        # train the model
        model.train()
        # initialize epoch train loss
        train_loss_mean = 0
        # loop over train batches
        for train_X, train_Y in train_loader:
            # compute loss for train
            train_output, train_loss = loss_function(model, train_X, train_Y, train_data)
            # add batch loss into epoch mean
            train_loss_mean += train_loss.item()*len(train_X) / len(train_loader.dataset)
            # clear old gradient
            model.zero_grad()
            # compute gradient
            train_loss.backward()
            # update model parameter
            optimizer.step()
        # evalute the model
        model.eval()
        # initialize epoch validation loss
        valid_loss_mean = 0
        # disable gradient during validation
        with torch.no_grad():
            # loop over validation batches
            for valid_X, valid_Y in validation_loader:
                # compute prediction and validation loss
                valid_output, valid_loss = loss_function(model, valid_X, valid_Y, train_data)
                # add batch loss into epoch mean
                valid_loss_mean += valid_loss.item()*len(valid_X) / len(validation_loader.dataset)
        # save epoch train and validation loss
        train_epoch_loss.append(train_loss_mean)
        valid_epoch_loss.append(valid_loss_mean)
        # print epoch result
        print(
            f"[{epoch + 1}/{num_epochs}] "
            f"Train Loss = {train_loss_mean:.4f}; "
            f"Valid Loss = {valid_loss_mean:.4f}"
        )
    return train_epoch_loss, valid_epoch_loss

# purpose: evaluate train model by test MSE and test perceptual loss
# output: result with test MSE and test perceptual loss
def evaluate_model_metrics(model, test_loader, train_data, loss_function):
    # evaluate the model
    model.eval()
    # initialize test MSE and test perceptual loss
    test_mse_mean = 0
    test_perceptual_loss_mean = 0
    # disable gradients during testing
    with torch.no_grad():
        # loop over test batches
        for test_X, test_Y in test_loader:
            # compute predict color image
            prediction = model(test_X)
            # compute MSE
            mse_value = nn.MSELoss()(prediction, test_Y)
            # compute perceptual loss
            _, perceptual_loss_value = loss_function(model, test_X, test_Y, train_data)
            # add MSE to dataset mean
            test_mse_mean += mse_value.item()*len(test_X) / len(test_loader.dataset)
            # add perceptual loss to dataset mean
            test_perceptual_loss_mean += perceptual_loss_value.item()*len(test_X) / len(test_loader.dataset)
    return {
        "test_mse": test_mse_mean,
        "test_perceptual_loss": test_perceptual_loss_mean
    }

# purpose: run hyperparameter choice
# output: model, test metrics, training curve, validation curve, test loader
# side effect: train model and print result
def run_one_experiment(choice_name, select_layer, max_layer, mse_weight):
    # build train, validation, and test dataloaders
    train_loader, validation_loader, test_loader = build_dataloaders(batch_size=batch_size)
    # load vgg19 model
    train_data = load_cifar_vgg19()
    # create UNet
    model = UNet()
    # create the loss
    loss_function = make_perceptual_loss(select_layer=select_layer, max_layer=max_layer, mse_weight=mse_weight)
    # train the model and store loss
    train_loss, valid_loss = train_one_model(model=model, train_loader=train_loader, validation_loader=validation_loader, 
                                             train_data=train_data, loss_function=loss_function)
    # evaluate the model on test set
    test_metrics = evaluate_model_metrics(model=model, test_loader=test_loader, train_data=train_data, loss_function=loss_function)
    # print experiment result
    print(
        f"{choice_name}: "
        f"select_layer={select_layer}, "
        f"max_layer={max_layer}, "
        f"mse_weight={mse_weight}, "
        f"test MSE={test_metrics['test_mse']:.4f}, "
        f"test perceptual loss={test_metrics['test_perceptual_loss']:.4f}, "
    )
    return model, test_metrics, train_loss, valid_loss, test_loader

# purpose: compare different vgg feature layers in perceptual loss
# output: result for metrics, model, curve, and test loader
# side effect: trains each layer experiment
def run_layer_comparison():
    # define layer choices for experiment
    layer_choice = {
        "texture_2_6": {"select_layer": (2, 6), "max_layer": 10, "mse_weight": 0.06},
        "structure_7_10": {"select_layer": (7, 10), "max_layer": 11, "mse_weight": 0.06},
        "semantic_12_14": {"select_layer": (12, 14), "max_layer": 15, "mse_weight": 0.06},
        "multi_2_6_10": {"select_layer": (2, 6, 10), "max_layer": 11, "mse_weight": 0.06}
    }
    results = {}
    models = {}
    curves = {}
    # loop over all layer choices
    for choice_name, choice in layer_choice.items():
        print(f"\nRunning layer comparison: {choice_name}")
        # train and evaluate one layer
        model, test_metrics, train_loss, valid_loss, test_loader = run_one_experiment(choice_name=choice_name, select_layer=choice["select_layer"], 
                                                                                      max_layer=choice["max_layer"], mse_weight=choice["mse_weight"])
        # save test metrics, model, and loss curve
        results[choice_name] = test_metrics
        models[choice_name] = model
        curves[choice_name] = {"train": train_loss, "valid": valid_loss}
    return results, models, curves, test_loader

# purpose: compare different MSE weights for multi layers
# output: result for metrics, models, curves, and test loader
# side effect: trains each MSE experiment
def run_mse_weight_comparison():
    # define MSE choices for experiment
    mse_weight_choice = {
        "mse_weight_0": {"select_layer": (2, 6, 10), "max_layer": 11, "mse_weight": 0.00},
        "mse_weight_6": {"select_layer": (2, 6, 10), "max_layer": 11, "mse_weight": 0.06},
        "mse_weight_20": {"select_layer": (2, 6, 10), "max_layer": 11, "mse_weight": 0.20},
        "mse_weight_40": {"select_layer": (2, 6, 10), "max_layer": 11, "mse_weight": 0.40}
    }
    results = {}
    models = {}
    curves = {}
    # loop over all MSE weight choices
    for choice_name, choice in mse_weight_choice.items():
        print(f"\nRunning MSE comparison: {choice_name}")
        # train and evaluate one weight
        model, test_metrics, train_loss, valid_loss, test_loader = run_one_experiment(choice_name=choice_name, select_layer=choice["select_layer"],
                                                                                      max_layer=choice["max_layer"], mse_weight=choice["mse_weight"])
        # save test metrics, model, and loss curve
        results[choice_name] = test_metrics
        models[choice_name] = model
        curves[choice_name] = {"train": train_loss, "valid": valid_loss}
    return results, models, curves, test_loader

# purpose: save image of different model output
# side effect: save figure to disk
def plot_qualitative_figure(model_lst, test_loader, model_keys, image_index, title, save_path):
    # get one batch from the test set
    grayscale_batch, color_batch = next(iter(test_loader))
    # select one grayscale image and keep batch dimension
    grayscale_image = grayscale_batch[image_index:image_index + 1]
    # create figure with input, target, and prediction
    plt.figure(figsize=(4*(2+len(model_keys)), 4))
    # create subplot for grayscale input
    plt.subplot(1, 2+len(model_keys), 1)
    # display grayscale input image
    plt.imshow(grayscale_batch[image_index].squeeze(), cmap="gray")
    plt.title("Input grayscale")
    plt.axis("off")
    # create subplot for target color image
    plt.subplot(1, 2+len(model_keys), 2)
    # display target image
    plt.imshow(color_batch[image_index].permute(1, 2, 0).clamp(0, 1))
    plt.title("Ground truth")
    plt.axis("off")
    # loop over train models to show prediction
    for column_index, key in enumerate(model_keys, start=3):
        # get model by key
        model = model_lst[key]
        # evaluate the model
        model.eval()
        # disable gradients for prediction
        with torch.no_grad():
            # compute prediction
            prediction = model(grayscale_image)[0]
        # create subplot for model prediction
        plt.subplot(1, 2+len(model_keys), column_index)
        # display prediction
        plt.imshow(prediction.permute(1, 2, 0).clamp(0, 1))
        plt.title(key)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# purpose: save quantitative table for hyperparameter experiment
# side effect: save figure to disk
def save_summary_table(layer_results, mse_weight_results, save_path):
    # create figure and axis
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    table_data = []
    # loop over layer result
    for choice_name, metrics in layer_results.items():
        table_data.append(["Feature layer choice", choice_name, f"test MSE = {metrics['test_mse']:.4f}, test perceptual loss = {metrics['test_perceptual_loss']:.4f}"])
        # loop over MSE result
    for choice_name, metrics in mse_weight_results.items():
        table_data.append(["MSE coefficient", choice_name, f"test MSE = {metrics['test_mse']:.4f}, test perceptual loss = {metrics['test_perceptual_loss']:.4f}"])
    # draw table
    column_labels = ["experiment", "choice", "metrics"]
    ax.table(cellText=table_data, colLabels=column_labels, loc="center",cellLoc="center")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# purpose: run hyperparameter experiments and save figures
# side effect: trains models and saves figures to current directory
def main():
    # run feature layer experiment
    layer_results, layer_models, layer_curves, layer_test_loader = run_layer_comparison()
    # run MSE experiment
    mse_weight_results, mse_weight_models, mse_weight_curves, mse_weight_test_loader = run_mse_weight_comparison()
    # save quantitative result table
    save_summary_table(layer_results=layer_results, mse_weight_results=mse_weight_results, save_path="perceptual_hyperparameter_table.png")
    # save qualitative result figure for layer comparison
    plot_qualitative_figure(model_lst=layer_models, test_loader=layer_test_loader, model_keys=["texture_2_6", "structure_7_10", "semantic_12_14", "multi_2_6_10"],
        image_index=0, title="Feature layers comparison figure", save_path="perceptual_layer_qualitative.png")
    # save qualitative result figure for MSE comparison
    plot_qualitative_figure(model_lst=mse_weight_models, test_loader=mse_weight_test_loader, model_keys=["mse_weight_0", "mse_weight_6", "mse_weight_20", "mse_weight_40"], 
        image_index=0, title="MSE weights comparison figure", save_path="mse_comparison_qualitative.png")

if __name__ == "__main__":
    main()
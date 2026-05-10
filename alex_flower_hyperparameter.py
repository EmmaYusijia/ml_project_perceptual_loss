from data import build_flowers_dataloaders
from UNet import UNet
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
import copy
import torchvision.models as models

num_epochs = 10
learning_rate = 0.001
batch_size = 16
image_size = 64

# purpose: load the AlexNet model
# output: AlexNet model
# side effect: download the model
def load_AlexNet():
    # load the AlexNet model
    train_data = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    # change the ReLU
    for layer in train_data.features:
        if isinstance(layer, nn.ReLU):
            layer.inplace = False
    # put the model in eval as we do not train it
    train_data.eval()
    # remove the parameter in the features
    for param in train_data.parameters():
        param.requires_grad = False
    return train_data

# purpose: normalize image based on AlexNet mean and standard deviation
# output: normalized image for AlexNet
def normalize_for_AlexNet(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image = (image-mean)/std
    return image

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
        predict_feature = normalize_for_AlexNet(prediction.clamp(0, 1))
        target_feature = normalize_for_AlexNet(Y.clamp(0, 1))
        # store perceptual loss of one choice
        feature_loss_lst = []
        # add prediction and target into AlexNet model
        for layer_index in range(max_layer):
            # select one AlexNet layer choice
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
# output: model, training curve, validation curve, epoch
# side effect: update model parameter
def train_one_model(model, train_loader, validation_loader, train_data, loss_function, learning_rate):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # store mean train loss and validation loss for each epoch
    train_epoch_loss = []
    valid_epoch_loss = []
    # store model outputs at different epochs
    epoch_model = {}
    # choose epochs to save qualitative results
    save_epochs = [1, 5, 7, 10]
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
        # save model at a specific epoch for qualitative comparison
        if epoch + 1 in save_epochs:
            epoch_model[epoch+1] = copy.deepcopy(model)
        # print epoch result
        print(
            f"[{epoch + 1}/{num_epochs}]"
            f"Train Loss = {train_loss_mean:.4f};"
            f"Valid Loss = {valid_loss_mean:.4f}"
        )
    return train_epoch_loss, valid_epoch_loss, epoch_model

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
            test_output, test_perceptual_loss = loss_function(model, test_X, test_Y, train_data)
            # add MSE to dataset mean
            test_mse_mean += mse_value.item()*len(test_X) / len(test_loader.dataset)
            # add perceptual loss to dataset mean
            test_perceptual_loss_mean += test_perceptual_loss.item()*len(test_X) / len(test_loader.dataset)
    return {"test_mse": test_mse_mean, "test_perceptual_loss": test_perceptual_loss_mean}

# purpose: run hyperparameter choice
# output: model, test metrics, training curve, validation curve, test loader, epoch
# side effect: train model and print result
def run_one_experiment(choice_name, select_layer, max_layer, mse_weight, learning_rate):
    # select one seed
    torch.manual_seed(42)
    # build train, validation, and test dataloaders
    train_loader, validation_loader, test_loader = build_flowers_dataloaders(batch_size=batch_size, image_size=image_size)
    # load AlexNet model
    train_data = load_AlexNet()
    # create UNet
    model = UNet()
    # create the loss
    loss_function = make_perceptual_loss(select_layer=select_layer, max_layer=max_layer, mse_weight=mse_weight)
    # train the model and store loss
    train_loss, valid_loss, epoch_model = train_one_model(model=model, train_loader=train_loader, validation_loader=validation_loader, 
                                             train_data=train_data, loss_function=loss_function, learning_rate=learning_rate)
    # evaluate the model on test set
    test_metrics = evaluate_model_metrics(model=model, test_loader=test_loader, train_data=train_data, loss_function=loss_function)
    # print experiment result
    print(
        f"{choice_name}: "
        f"select_layer={select_layer}, "
        f"max_layer={max_layer}, "
        f"mse_weight={mse_weight}, "
        f"learning_rate={learning_rate}, "
        f"test MSE={test_metrics['test_mse']:.4f}, "
        f"test perceptual loss={test_metrics['test_perceptual_loss']:.4f}, "
    )
    return model, test_metrics, train_loss, valid_loss, test_loader, epoch_model

# purpose: compare different AlexNet feature layers in perceptual loss
# output: result for metrics, model, curve, test loader, and epoch output
# side effect: trains each layer experiment
def run_layer_comparison():
    # define layer choices for experiment
    layer_choice = {
        "texture_0_5": {"select_layer": (0, 1, 2, 3, 4, 5), "max_layer": 6, "mse_weight": 0.05, "learning_rate": 0.0005},
        "structure_6_9": {"select_layer": (6, 7, 8, 9), "max_layer": 10, "mse_weight": 0.05, "learning_rate": 0.0005},
        "semantic_10_12": {"select_layer": (10, 11, 12), "max_layer": 13, "mse_weight": 0.05, "learning_rate": 0.0005},
        "multiple_0_12": {"select_layer": (0, 2, 5, 7, 9, 10), "max_layer": 11, "mse_weight": 0.05, "learning_rate": 0.0005}
    }
    result = {}
    models = {}
    # loop over all layer choices
    for choice_name, choice in layer_choice.items():
        print(f"\nRunning layer comparison: {choice_name}")
        # train and evaluate one layer
        model, test_metrics, train_loss, valid_loss, test_loader, epoch_model = run_one_experiment(choice_name=choice_name, select_layer=choice["select_layer"], 
                                                                                      max_layer=choice["max_layer"], mse_weight=choice["mse_weight"],
                                                                                      learning_rate=choice["learning_rate"])
        # save test metrics, model, loss curve, and epoch output
        result[choice_name] = test_metrics
        models[choice_name] = model
    return result, models, test_loader

# purpose: compare different MSE weights for multi layers
# output: result for metrics, models, curves, test loader, and epoch output
# side effect: trains each MSE experiment
def run_mse_weight_comparison():
    # define MSE choices for experiment
    mse_weight_choice = {
        "mse_weight_0": {"select_layer": (0, 2, 5, 7, 9, 10), "max_layer": 11, "mse_weight": 0.00, "learning_rate": 0.0005},
        "mse_weight_5": {"select_layer": (0, 2, 5, 7, 9, 10), "max_layer": 11, "mse_weight": 0.05, "learning_rate": 0.0005},
        "mse_weight_10": {"select_layer": (0, 2, 5, 7, 9, 10), "max_layer": 11, "mse_weight": 0.10, "learning_rate": 0.0005},
        "mse_weight_20": {"select_layer": (0, 2, 5, 7, 9, 10), "max_layer": 11, "mse_weight": 0.20, "learning_rate": 0.0005}
    }
    result = {}
    models = {}
    epoch_output = {}
    # loop over all MSE weight choices
    for choice_name, choice in mse_weight_choice.items():
        print(f"\nRunning MSE comparison: {choice_name}")
        # train and evaluate one weight
        model, test_metrics, train_loss, valid_loss, test_loader, epoch_model = run_one_experiment(choice_name=choice_name, select_layer=choice["select_layer"],
                                                                                      max_layer=choice["max_layer"], mse_weight=choice["mse_weight"],
                                                                                      learning_rate=choice["learning_rate"])
        # save test metrics, model, loss curve, and epoch output
        result[choice_name] = test_metrics
        models[choice_name] = model
        epoch_output[choice_name] = epoch_model
    return result, models, test_loader, epoch_output

# purpose: compare different learning rates for multi layers
# output: result for metrics, models, curves, test loader, and epoch output
# side effect: trains each learning rate experiment
def run_learning_rate_comparison():
    # define learning rate choices for experiment
    learning_rate_choice = {
        "learning_rate_0001": {"select_layer": (0, 2, 5, 7, 9, 10), "max_layer": 11, "mse_weight": 0.05, "learning_rate": 0.0001},
        "learning_rate_0005": {"select_layer": (0, 2, 5, 7, 9, 10), "max_layer": 11, "mse_weight": 0.05, "learning_rate": 0.0005},
        "learning_rate_001": {"select_layer": (0, 2, 5, 7, 9, 10), "max_layer": 11, "mse_weight": 0.05, "learning_rate": 0.001}
    }
    result = {}
    models = {}
    # loop over all learning rate choices
    for choice_name, choice in learning_rate_choice.items():
        print(f"\nRunning learning rate comparison: {choice_name}")
        # train and evaluate one learning rate
        model, test_metrics, train_loss, valid_loss, test_loader, epoch_model = run_one_experiment(choice_name=choice_name, select_layer=choice["select_layer"],
                                                                                      max_layer=choice["max_layer"], mse_weight=choice["mse_weight"],
                                                                                      learning_rate=choice["learning_rate"])
        # save test metrics, model, loss curve, and epoch output
        result[choice_name] = test_metrics
        models[choice_name] = model
    return result, models, test_loader

# purpose: get one example of five different flower labels of the test dataset
# output: list of grayscale images, color images, and labels
# side effect: None
def get_one_example(test_loader):
    # get the dataset
    colorization_data = test_loader.dataset
    flower_data = colorization_data.dataset
    grayscale_image_lst = []
    color_image_lst = []
    label_lst = []
    used_labels = set()
    # loop over the whole flower dataset
    for index in range(len(flower_data)):
        # get color image and label from flower dataset
        color_image, label = flower_data[index]
        # keep the first example for each label
        if label not in used_labels:
            grayscale_image = colorization_data.grayscale_transform(color_image)
            grayscale_image_lst.append(grayscale_image)
            color_image_lst.append(color_image)
            label_lst.append(label)
            used_labels.add(label)
        # stop after five different labels
        if len(label_lst) == 5:
            break
    return grayscale_image_lst, color_image_lst, label_lst

# purpose: save image of different model output
# side effect: save figure to disk
def plot_qualitative_figure(model_lst, test_loader, model_keys, image_count, title, save_path):
    # get one example from each flower label
    grayscale_image_lst, color_image_lst, label_lst = get_one_example(test_loader)
    # use the first label
    grayscale_image_lst = grayscale_image_lst[:image_count]
    color_image_lst = color_image_lst[:image_count]
    label_lst = label_lst[:image_count]
    # create figure with input, target, and prediction
    column_count = 2 + len(model_keys)
    plt.figure(figsize=(4*column_count, 4*image_count))
    # loop over images
    for row_index in range(image_count):
        # select one grayscale image and keep batch dimension
        grayscale_image = grayscale_image_lst[row_index].unsqueeze(0)
        # create subplot for grayscale input
        plt.subplot(image_count, column_count, row_index*column_count + 1)
        plt.imshow(grayscale_image_lst[row_index].squeeze(), cmap="gray")
        plt.title("Input grayscale\nlabel_" + str(label_lst[row_index]))
        plt.axis("off")
        # create subplot for target color image
        plt.subplot(image_count, column_count, row_index*column_count + 2)
        plt.imshow(color_image_lst[row_index].permute(1, 2, 0).clamp(0, 1))
        plt.title("Ground truth")
        plt.axis("off")
        # loop over models to show prediction
        for column_index, key in enumerate(model_keys, start=3):
            # get model by epoch
            model = model_lst[key]
            model.eval()
            with torch.no_grad():
                prediction = model(grayscale_image)[0]
            plt.subplot(image_count, column_count, row_index*column_count + column_index)
            # display prediction
            plt.imshow(prediction.permute(1, 2, 0).clamp(0, 1))
            plt.title(key)
            plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# purpose: save qualitative results at different epochs for one experiment
# side effect: save figure to disk
def plot_epoch_qualitative_figure(epoch_model, test_loader, image_count, title, save_path):
    # get one example from each flower label
    grayscale_image_lst, color_image_lst, label_lst = get_one_example(test_loader)
    # use the first label
    grayscale_image_lst = grayscale_image_lst[:image_count]
    color_image_lst = color_image_lst[:image_count]
    label_lst = label_lst[:image_count]
    # get epoch number
    model_keys = list(epoch_model.keys())
    # create figure with input, target, and prediction
    column_count = 2 + len(model_keys)
    plt.figure(figsize=(4*column_count, 4*image_count))
    # loop over images
    for row_index in range(image_count):
        # select one grayscale image and keep batch dimension
        grayscale_image = grayscale_image_lst[row_index].unsqueeze(0)
        # create subplot for grayscale input
        plt.subplot(image_count, column_count, row_index*column_count + 1)
        plt.imshow(grayscale_image_lst[row_index].squeeze(), cmap="gray")
        plt.title("Input grayscale\nlabel_" + str(label_lst[row_index]))
        plt.axis("off")
        # create subplot for target color image
        plt.subplot(image_count, column_count, row_index*column_count + 2)
        plt.imshow(color_image_lst[row_index].permute(1, 2, 0).clamp(0, 1))
        plt.title("Ground truth")
        plt.axis("off")
        # loop over epoch to show prediction
        for column_index, epoch_key in enumerate(model_keys, start=3):
            # get model by epoch
            model = epoch_model[epoch_key]
            model.eval()
            with torch.no_grad():
                prediction = model(grayscale_image)[0]
            plt.subplot(image_count, column_count, row_index*column_count + column_index)
            # display prediction
            plt.imshow(prediction.permute(1, 2, 0).clamp(0, 1))
            plt.title("epoch_" + str(epoch_key))
            plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# purpose: save quantitative table for hyperparameter experiment
# side effect: save figure to disk
def save_summary_table(layer_result, mse_weight_result, learning_rate_result, save_path):
    # create figure and axis
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    table_data = []
    # loop over layer result
    for choice_name, metrics in layer_result.items():
        table_data.append(["Feature layer choice", choice_name, f"test MSE = {metrics['test_mse']:.4f}"])
        # loop over MSE result
    for choice_name, metrics in mse_weight_result.items():
        table_data.append(["MSE coefficient", choice_name, f"test MSE = {metrics['test_mse']:.4f}, test perceptual loss = {metrics['test_perceptual_loss']:.4f}"])
    # loop over learning rate result
    for choice_name, metrics in learning_rate_result.items():
        table_data.append(["Learning rate", choice_name, f"test MSE = {metrics['test_mse']:.4f}, test perceptual loss = {metrics['test_perceptual_loss']:.4f}"])
    # draw table
    column_labels = ["experiment", "choice", "metrics"]
    ax.table(cellText=table_data, colLabels=column_labels, loc="center",cellLoc="center")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# purpose: run hyperparameter experiments and save figures
# side effect: trains models and saves figures to current directory
def main():
    # run feature layer experiment
    layer_result, layer_model, layer_test_loader = run_layer_comparison()
    # run MSE experiment
    mse_weight_result, mse_weight_model, mse_weight_test_loader, mse_weight_epoch_model = run_mse_weight_comparison()
    # run learning rate experiment
    learning_rate_result, learning_rate_model, learning_rate_test_loader = run_learning_rate_comparison()
    # save quantitative result table
    save_summary_table(layer_result=layer_result, mse_weight_result=mse_weight_result, learning_rate_result=learning_rate_result,
                       save_path="AlexNet_flower_perceptual_hyperparameter_table.png")
    # save qualitative result figure for layer comparison
    plot_qualitative_figure(model_lst=layer_model, test_loader=layer_test_loader, model_keys=["texture_0_5", "structure_6_9", "semantic_10_12", "multiple_0_12"],
        image_count=5, title="AlexNet flower feature layers comparison figure", save_path="AlexNet_flower_perceptual_layer_qualitative.png")
    # save qualitative result figure for MSE comparison
    plot_qualitative_figure(model_lst=mse_weight_model, test_loader=mse_weight_test_loader, model_keys=["mse_weight_0", "mse_weight_5", "mse_weight_10", "mse_weight_20"], 
        image_count=5, title="AlexNet flower MSE weights comparison figure", save_path="AlexNet_flower_mse_comparison_qualitative.png")
    # save qualitative result figure for learning rate comparison
    plot_qualitative_figure(model_lst=learning_rate_model, test_loader=learning_rate_test_loader, model_keys=["learning_rate_0001", "learning_rate_0005", "learning_rate_001"], 
        image_count=5, title="AlexNet flower learning rate comparison figure", save_path="AlexNet_flower_learning_rate_qualitative.png")
    # save epoch qualitative result figure for one experiment with learning rate = 0.001, layers = 2, 6, 10, 12, weights = 0.2
    plot_epoch_qualitative_figure(epoch_model=mse_weight_epoch_model["mse_weight_5"], test_loader=mse_weight_test_loader, image_count=5,
        title="Flower qualitative results at different epochs", save_path="AlexNet_flower_epoch_qualitative.png")

if __name__ == "__main__":
    main()
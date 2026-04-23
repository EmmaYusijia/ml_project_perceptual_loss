from fastprogress.fastprogress import master_bar, progress_bar
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device.")

# Mini-Batch SGD hyperparameters
num_epochs = 5
lr = 0.001

def mse_loss(model, X, Y):
    # Compute the output
    train_output = model(X)
    criterion = nn.MSELoss()
    # Compute loss
    train_loss = criterion(train_output, Y)
    return train_output, train_loss

def mae_loss(model, X, Y):
    train_output = model(X)
    criterion = nn.L1Loss()
    train_loss = criterion(train_output, Y)
    return train_output, train_loss

def run_model(model, training_loader, validation_loader, 
              optimizer, learning_rate, get_output_and_loss):    
    return gradient_descent(model, training_loader, validation_loader, optimizer, learning_rate, get_output_and_loss)

def visualize(model, model_name, test_loader):
    grayscale_batch, color_batch = next(iter(test_loader))
    grayscale_small = grayscale_batch[0:3, :, :, :]
    color_small = color_batch[0:3, :, :, :]

    model.eval()
    with torch.no_grad():
        outputs = model(grayscale_small)

    for i in range(outputs.shape[0]):
        input_img = grayscale_small[i].detach().cpu().permute(1, 2, 0).clamp(0,1)
        pred_img = outputs[i].detach().cpu().permute(1, 2, 0).clamp(0,1)
        true_img = color_small[i].detach().cpu().permute(1, 2, 0).clamp(0,1)
    
        # input
        plt.title(f"Input{i}")
        plt.imshow(input_img) 
        plt.savefig(model_name + f"Input{i}")

        # prediction
        plt.title(f"Prediction{i}")
        plt.imshow(pred_img)
        plt.savefig(model_name + f"Prediction{i}")

        # target
        plt.title(f"Target{i}")
        plt.imshow(true_img)
        plt.savefig(model_name + f"Target{i}")

def gradient_descent(model, train_loader, valid_loader, optimizer, learning_rate, get_output_and_loss):

    # Do model creation here so that the model is recreated each time the cell is run
    model = model.to(device)

    t = 0
    # Create the optimizer, just like we have with the built-in optimizer
    opt = optimizer(model.parameters(), learning_rate)

    # A master bar for fancy output progress
    mb = master_bar(range(num_epochs))

    # Information for plots
    mb.names = ["Train Loss", "Valid Loss"]
    train_losses = []
    valid_losses = []

    for epoch in mb:

        #
        # Training
        #
        model.train()

        train_N = len(train_loader.dataset)
        num_train_batches = len(train_loader)
        train_dataiterator = iter(train_loader)

        train_loss_mean = 0

        for batch in progress_bar(range(num_train_batches), parent=mb):

            # Grab the batch of data and send it to the correct device
            train_X, train_Y = next(train_dataiterator)
            train_X, train_Y = train_X.to(device), train_Y.to(device)

            train_output, train_loss = get_output_and_loss(model, train_X, train_Y)

            num_in_batch = len(train_X)
            tloss = train_loss.item() * num_in_batch / train_N
            train_loss_mean += tloss
            train_losses.append(train_loss.item())

            # Compute gradient
            model.zero_grad()
            train_loss.backward()
            
            # Take a step of gradient descent
            t += 1
            with torch.no_grad():
                opt.step()

        #
        # Validation
        #
        model.eval()

        valid_N = len(valid_loader.dataset)
        num_valid_batches = len(valid_loader)

        valid_loss_mean = 0
        valid_correct = 0

        with torch.no_grad():

            # valid_loader is probably just one large batch, so not using progress bar
            for valid_X, valid_Y in valid_loader:

                valid_X, valid_Y = valid_X.to(device), valid_Y.to(device)

                valid_output, valid_loss = get_output_and_loss(model, valid_X, valid_Y)

                num_in_batch = len(valid_X)
                vloss = valid_loss.item() * num_in_batch / valid_N
                valid_loss_mean += vloss
                valid_losses.append(valid_loss.item())

                try:
                    # Convert network output into predictions (one-hot -> number)
                    predictions = valid_output.argmax(1)

                    # Sum up total number that were correct
                    valid_correct += (predictions == valid_Y).type(torch.float).sum().item()
                except:
                    pass

        valid_accuracy = 100 * (valid_correct / valid_N)

        # Report information
        tloss = f"Train Loss = {train_loss_mean:.4f}"
        vloss = f"Valid Loss = {valid_loss_mean:.4f}"
        vaccu = f"Valid Accuracy = {(valid_accuracy):>0.1f}%"
        mb.write(f"[{epoch+1:>2}/{num_epochs}] {tloss}; {vloss}; {vaccu}")

        # Update plot data
        max_loss = max(max(train_losses), max(valid_losses))
        min_loss = min(min(train_losses), min(valid_losses))

        x_margin = 0.2
        x_bounds = [0 - x_margin, num_epochs + x_margin]

        y_margin = 0.1
        y_bounds = [min_loss - y_margin, max_loss + y_margin]

        valid_Xaxis = torch.linspace(0, epoch + 1, len(train_losses))
        valid_xaxis = torch.linspace(1, epoch + 1, len(valid_losses))
        graph_data = [[valid_Xaxis, train_losses], [valid_xaxis, valid_losses]]

        mb.update_graph(graph_data, x_bounds, y_bounds)

    print(f"[{epoch+1:>2}/{num_epochs}] {tloss}; {vloss}; {vaccu}")
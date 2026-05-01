import matplotlib.pyplot as plt

def plot_loss_comp(epochs, a_train_loss, a_valid_loss, b_train_loss, b_valid_loss, label1, label2, label3, label4):
    plt.plot(epochs, a_train_loss, color='blue', linestyle='-', label=label1)
    plt.plot(epochs, a_valid_loss, color='blue', linestyle='--', label=label2)

    plt.plot(epochs, b_train_loss, color='orange', linestyle='-', label=label3)
    plt.plot(epochs, b_valid_loss, color='orange', linestyle='--', label=label4)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train vs Validation Comparison")

    plt.legend()
    plt.show()

if __name__ == '__main__':
    epochs = list(range(1, 11))
    baseline_train = [0.2025, 0.1723, 0.1672, 0.1623, 0.1603, 0.1570, 0.1552, 0.1523, 0.1520, 0.1503]
    baseline_valid = [0.1720, 0.1673, 0.1617, 0.1550, 0.1578, 0.1542, 0.1513, 0.1504, 0.1521, 0.1469]
    wo_extract_train = [0.2782, 0.2266, 0.2222, 0.2206, 0.2190, 0.2178, 0.2153, 0.2146, 0.2143, 0.2141]
    wo_extract_valid = [0.2265, 0.2213, 0.2178, 0.2189, 0.2151, 0.2129, 0.2168, 0.2116, 0.2123, 0.2120]

    # plot_loss_comp(epochs, UNet_train_mse_loss, UNet_valid_mse_loss, AE_train_mse_loss, AE_valid_mse_loss, 
    #                "UNet Train", "UNet Valid", "AE Train", "AE Valid", "MSE Loss")
    # plot_loss_comp(epochs, UNet_train_mae_loss, UNet_valid_mae_loss, AE_train_mae_loss, AE_valid_mae_loss, 
    #                "UNet Train", "UNet Valid", "AE Train", "AE Valid", "MAE Loss")
    plot_loss_comp(epochs, baseline_train, baseline_valid, wo_extract_train, wo_extract_valid, 
                   "intermediate train", "intermediate valid", "output train", "output valid")

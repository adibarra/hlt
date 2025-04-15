import matplotlib.pyplot as plt

# plot for FFNN
epochs = list(range(1, 14))
train_loss = [1.4972, 1.2299, 1.0248, 0.8670, 0.7339, 0.6308, 0.5347, 0.4558, 0.3956, 0.3377, 0.2931, 0.2671, 0.2460]
val_acc = [0.5175, 0.5700, 0.5850, 0.5887, 0.5475, 0.5913, 0.5750, 0.5463, 0.5587, 0.5613, 0.5550, 0.5650, 0.5613]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = "tab:blue"
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss", color=color)
ax1.plot(epochs, train_loss, color=color, label="Train Loss", linewidth=2)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "tab:green"
ax2.set_ylabel("Validation Accuracy", color=color)
ax2.plot(epochs, val_acc, color=color, label="Val Accuracy", linewidth=2, linestyle="--")
ax2.tick_params(axis="y", labelcolor=color)

fig.subplots_adjust(top=0.92)
plt.title("Learning Curve: FFNN (2-layer, 16 hidden units, LR=3e-4)", fontsize=14)
plt.grid(visible=True)
plt.show()

# plot for RNN
epochs = list(range(1, 31))
train_loss = [1.5573, 1.4746, 1.4531, 1.4331, 1.4271, 1.4066, 1.4004, 1.3846, 1.3674, 1.3714, 1.3568, 1.3508, 1.3131, 1.3117, 1.2989, 1.2909, 1.2868, 1.2789, 1.2738, 1.2515, 1.2442, 1.2399, 1.2342, 1.2308, 1.2246, 1.2255, 1.2097, 1.2119, 1.2061, 1.2082]
val_acc = [0.3362, 0.3812, 0.2637, 0.3638, 0.3688, 0.4012, 0.4575, 0.4688, 0.4450, 0.4662, 0.4612, 0.4138, 0.4738, 0.4988, 0.5012, 0.3463, 0.4650, 0.4612, 0.4325, 0.4575, 0.3475, 0.5288, 0.4550, 0.5262, 0.4875, 0.5487, 0.4525, 0.4338, 0.4088, 0.4325]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = "tab:blue"
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss", color=color)
ax1.plot(epochs, train_loss, color=color, label="Train Loss", linewidth=2)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "tab:green"
ax2.set_ylabel("Validation Accuracy", color=color)
ax2.plot(epochs, val_acc, color=color, label="Val Accuracy", linewidth=2, linestyle="--")
ax2.tick_params(axis="y", labelcolor=color)

fig.subplots_adjust(top=0.92)
plt.title("Learning Curve: RNN (2-layer, 32 hidden units, LR=3e-4)", fontsize=14)
plt.grid(visible=True)
plt.show()

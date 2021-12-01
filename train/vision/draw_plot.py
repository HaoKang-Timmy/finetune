import matplotlib.pyplot as plt
import numpy as np
iter_list = []
loss_train_list = []
acc_train_list = []
loss_val_list = []
acc_val_list = []
with open('../log/train_bias.txt', 'r') as file:
    for line in file.readlines():
        line = line.strip().split("  ")
        iter, loss_train, acc_train, loss_val, acc_val = line[:]
        iter_list.append(int(iter.split(':')[1]))
        loss_train_list.append(float(loss_train.split(':')[1]))
        acc_train_list.append(float(acc_train.split(':')[1]))
        loss_val_list.append(float(loss_val.split(':')[1]))
        acc_val_list.append(float(acc_val.split(':')[1]))
ax1 = plt.subplot(2, 2, 1, frameon=False)
plt.plot(iter_list, loss_train_list)
ax2 = plt.subplot(2, 2, 2, frameon=False)
plt.plot(iter_list, acc_train_list)
ax3 = plt.subplot(2, 2, 3, frameon=False)
plt.plot(iter_list, loss_val_list)
ax4 = plt.subplot(2, 2, 4, frameon=False)
plt.plot(iter_list, acc_val_list)
plt.show()

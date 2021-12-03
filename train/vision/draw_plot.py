import matplotlib.pyplot as plt
import numpy as np
iter_list = []
loss_train_list = []
acc_train_list = []
loss_val_list = []
acc_val_list = []

loss_train_list1 = []
acc_train_list1 = []
loss_val_list1 = []
acc_val_list1 = []

loss_train_list2 = []
acc_train_list2 = []
loss_val_list2 = []
acc_val_list2 = []
with open('../log/train_bias.txt', 'r') as file:
    for line in file.readlines():
        line = line.strip().split("  ")
        iter, loss_train, acc_train, loss_val, acc_val = line[:]
        iter_list.append(int(iter.split(':')[1]))
        loss_train_list.append(float(loss_train.split(':')[1]))
        acc_train_list.append(float(acc_train.split(':')[1]))
        loss_val_list.append(float(loss_val.split(':')[1]))
        acc_val_list.append(float(acc_val.split(':')[1]))
with open('../log/train.txt', 'r') as file:
    for line in file.readlines():
        line = line.strip().split("  ")
        iter, loss_train1, acc_train1, loss_val1, acc_val1 = line[:]
        loss_train_list1.append(float(loss_train1.split(':')[1]))
        acc_train_list1.append(float(acc_train1.split(':')[1]))
        loss_val_list1.append(float(loss_val1.split(':')[1]))
        acc_val_list1.append(float(acc_val1.split(':')[1]))

with open('../log/train_tl_adam.txt', 'r') as file:
    for line in file.readlines():
        line = line.strip().split("  ")
        iter, loss_train2, acc_train2, loss_val2, acc_val2 = line[:]
        loss_train_list2.append(float(loss_train2.split(':')[1]))
        acc_train_list2.append(float(acc_train2.split(':')[1]))
        loss_val_list2.append(float(loss_val2.split(':')[1]))
        acc_val_list2.append(float(acc_val2.split(':')[1]))
ax1 = plt.subplot(2, 2, 1, frameon=True)
plt.plot(iter_list, loss_train_list, label="last")
plt.plot(iter_list, loss_train_list1, label="bias+last")
plt.plot(iter_list, loss_train_list2, label='tintl')
plt.title('training_loss')
plt.subplots_adjust(0.1, 0.1, 0.9, 0.9)
plt.legend()
ax2 = plt.subplot(2, 2, 2, frameon=True)
plt.plot(iter_list, acc_train_list, label="last")
plt.plot(iter_list, acc_train_list1, label="bias+last")
plt.plot(iter_list, acc_train_list2, label='tintl')
plt.title('training_acc(top1)')
plt.legend()
ax3 = plt.subplot(2, 2, 3, frameon=True)
plt.plot(iter_list, loss_val_list, label="last")
plt.plot(iter_list, loss_val_list1, label="bias+last")
plt.plot(iter_list, loss_val_list2, label='tintl')
plt.title('val_loss')
plt.legend()
ax4 = plt.subplot(2, 2, 4, frameon=True)
plt.plot(iter_list, acc_val_list, label="last")
plt.plot(iter_list, acc_val_list1, label="bias+last")
plt.plot(iter_list, acc_val_list2, label='tintl')
plt.title('val_acc(top1)')
plt.legend()
plt.tight_layout()
plt.show()

#!/usr/bin/env python3

import pickle as pickle
import numpy as np
import matplotlib.pyplot as plt


def predict(X, w, y=None, decay=0.0):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

    # TODO: Your code here
    y_hat = X @ w
    loss = (1/(2 * X.shape[0])) * ((X @ w - y)**2).sum() + decay * sum(w**2) 
    risk = (1/X.shape[0]) * sum(np.abs(X @ w - y))

    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val, decay, printing=False):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val = []

    w_best = None
    risk_best = 10000
    epoch_best = 0
    loss_best = 10000

    for epoch in range(MaxIter):

        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch, decay)
            loss_this_epoch += loss_batch

            gradient = lambda: ((1 / X_batch.shape[0]) * X_batch.T @ (X_batch @ w - y_batch) + 2 * decay * w)

            # TODO: Your code here
            # Mini-batch gradient descent
            w = w - alpha * gradient()
        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        # 2. Perform validation on the validation set by the risk
        # 3. Keep track of the best validation epoch, risk, and the weights
        
        ## Compute Training Loss
        loss = loss_this_epoch / batch_size
        losses_train.append(loss)

        ## Validation
        validation_risk = (1 / X_val.shape[0]) * sum(np.abs(X_val @ w - y_val))
        risks_val.append(validation_risk)
        if printing:
            print(f"Loss: {loss[0]:.3f}, Risk: {validation_risk[0]:.3f}")

        ## Validation best
        if validation_risk < risk_best:
            risk_best = validation_risk
            epoch_best = epoch
            w_best = w
            loss_best = loss

        risk_best = min(risks_val)

    # Return some variables as needed
    return losses_train, risks_val, risk_best, epoch_best, w, loss_best


############################
# Main code starts here
############################
# Load data
with open("housing.pkl", "rb") as f:
    (X, y) = pickle.load(f)

# X: sample x dimension
# y: sample x 1

X = np.concatenate((X, X**2), axis=1)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Augment feature
X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
# X_: Nsample x (d+1)

# normalize features:
mean_y = np.mean(y)
std_y = np.std(y)

y = (y - np.mean(y)) / np.std(y)

# print(X.shape, y.shape) # It's always helpful to print the shape of a variable


# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

print(X_.shape)
X_train = X_[:300]
y_train = y[:300]

X_val = X_[300:400]
y_val = y[300:400]

X_test = X_[400:]
y_test = y[400:]

#####################
# setting

alpha = 0.001      # learning rate
batch_size = 10    # batch size
MaxIter = 100        # Maximum iteration
best_decay = 8000         # weight decay
best_risk = 8000

# TODO: Your code here
for decay in [0.3]: #[3, 1, 0.3, 0.1, 0.03, 0.01]:
    loss, risk, brisk, bepoch, w, new = train(X_train, y_train, X_val, y_val, decay)
    if brisk < best_risk:
        best_risk = brisk
        best_decay = decay
        print(f"Best risk: {best_risk}, Best decay: {best_decay}")


print(w)
plt.close()
plt.plot(np.arange(len(loss)), loss)
plt.savefig('q2b_train_loss.png')
plt.close()
plt.plot(np.arange(len(risk)), risk)
plt.savefig('q2b_val_risk.png')
plt.close()

print(f"Best risk: {brisk}, Best epoch: {bepoch}")
_, _, test_risk = predict(X_test, w, y_test)
print(f"Test risk: {test_risk}")
    
    # Perform test by the weights yielding the best validation performance
    
    # Report numbers and draw plots as required.

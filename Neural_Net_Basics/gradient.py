import pandas as pd
from autograd import grad 
import autograd.numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def get_data_scaled(path):
    data = pd.read_csv(path)
    x = data[['x1','x2','x3']].values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    y = data['y'].values
    return x_scaled, y, data

def get_data_original(path):
    data = pd.read_csv(path)
    x = data[['x1','x2','x3']].values
    y = data['y'].values
    return x, y, data

def model(w,x,c):
    return np.dot(x,w)+c

def loss(w,x,c,y):
    y_pred = model(w, x, c)
    residuals = y_pred - y
    result = np.mean(residuals ** 2)
    return result

def compute_gradient(w, x, c, y):
    grad_loss_w = grad(loss, argnum=0) #compute the gradient function respect to w 
    grad_loss_c = grad(loss, argnum=2) #compute the gradient function respect to c
    grad_w = grad_loss_w(w, x, c, y) #put numbers in to compute gradient value for w
    grad_c = grad_loss_c(w, x, c, y) #put numbers in to compute gradient value for c
    return grad_w, grad_c

def update_momentum(w_initial, x, c_initial, y, learning_rate=0.1, epoch=100, momentum = 0.1):
    w_history = [w_initial]
    c_history = [c_initial]
    velocity_w = 0
    velocity_c = 0
    cost_history = [loss(w_initial,x,c_initial,y)] 
    w = w_initial
    c = c_initial

    for i in range(epoch):
        grad_w, grad_c = compute_gradient(w, x, c, y)

        if abs(grad_w[0]) <= 0.001 or abs(grad_c) <= 0.001:
            break

        velocity_w = velocity_w * momentum - learning_rate * grad_w
        velocity_c = velocity_c * momentum - learning_rate * grad_c

        w = w + velocity_w * momentum - learning_rate * grad_w
        c = c + velocity_c * momentum - learning_rate * grad_c

        w_history.append(w)
        c_history.append(c)
        cost_history.append(loss(w,x,c,y))

    return w_history, c_history, cost_history, w, c


def update_constantlearningrate(w_initial, x, c_initial, y, learning_rate=0.1, epoch=100):
    w_history = [w_initial]
    c_history = [c_initial]
    cost_history = [loss(w_initial,x,c_initial,y)] 
    w = w_initial
    c = c_initial

    for i in range(epoch):
        grad_w, grad_c = compute_gradient(w, x, c, y)

        if abs(grad_w[0]) <= 0.001 or abs(grad_c) <= 0.001:
            break

        w = w - learning_rate * grad_w
        c = c - learning_rate * grad_c

        w_history.append(w)
        c_history.append(c)
        cost_history.append(loss(w,x,c,y))

    return w_history, c_history, cost_history, w, c

def update_dynamiclearningrate(w_initial, x, c_initial, y, epoch=100):
    w_history = [w_initial]
    c_history = [c_initial]
    cost_history = [loss(w_initial,x,c_initial,y)] 
    w = w_initial
    c = c_initial

    for i in range(epoch):
        grad_w, grad_c = compute_gradient(w, x, c, y)

        if abs(grad_w[0]) <= 0.001 or abs(grad_c) <= 0.001:
            break

        learning_rate = (i+1)/epoch

        w = w - learning_rate * grad_w
        c = c - learning_rate * grad_c

        w_history.append(w)
        c_history.append(c)
        cost_history.append(loss(w,x,c,y))

    return w_history, c_history, cost_history, w, c

def plot(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Loss per Epoch', color='blue', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_learningrate(w, x, c, y):
    _, _, cost_history_constant, _, _ = update_constantlearningrate(w, x, c, y, learning_rate=0.1, epoch=100)
    _, _, cost_history_dynamic, _, _ = update_dynamiclearningrate(w, x, c, y, epoch=100)
    _, _, cost_history_momentum, _, _ = update_momentum(w, x, c, y, epoch=100)
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history_constant, label='Constant Learning Rate', color='blue', marker='o')
    plt.plot(cost_history_dynamic, label='Dynamic Learning Rate', color='red', marker='o')
    plt.plot(cost_history_momentum, label='Momentum Learning Rate', color='yellow', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__=='__main__':
    # x, y, df = get_data_original('gradient/data_nonstable.csv')
    x, y, df = get_data_scaled('gradient/data_nonstable.csv')
    w = np.random.randn(3)
    c = np.random.randn()

    # w_history, c_history, cost_history, w, c = update_constantlearningrate(w, x, c, y, learning_rate=0.1, epoch=100)
    # w_history, c_history, cost_history, w, c = update_dynamiclearningrate(w, x, c, y, epoch=100)
    # plot(cost_history)

    compare_learningrate(w, x, c, y)
    
    # y_pred = model(w,x,c)
    # df['y_pred'] = y_pred
    # print(df.head(200))

    
    


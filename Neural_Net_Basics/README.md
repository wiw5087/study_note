## Gradient.py
### Things that I learned
1. **How to code linear regression loss function in Python**: for linear regression $y=w_1*x_1 + w_2*x2 + w_3*x3 + c$, the loss function is MSE, which is $L = \frac 1 n * \sum(y_{pred} - y)^2$. In Python, this should be expressed as `np.mean(y_pred - y)`

2. **What's the partial derivative function**: Note that in Gradient Descent formula, we take the partial derivative of the Loss function against the parameter $\frac {dL} {dw}$, not $\frac {dy} {dw}$. 

3. **What's the update formula at each step**: Also notice that $value_{new} = value_{old} - LearningRate * delta$, don't forget the learning rate. 

4. **What are the two steps to use autograd**: we can manually work out the gradient by coding the functions. However we can also use `autograd` library to get the gradient function for w by doing `grad_loss_w = grad(loss, argnum=0)` assume loss function is `loss(w,x,c,y)`. Then we can feed in the actual value to get epoch value for gradient update by doing `grad_w = grad_loss_w(w,x,c,y)`. Notice that because we have 3 w, the grad_w result is an array with 3 values corresponding to each w `array([-0.89114301,  0.6723172 ,  0.01575687])`. Also notice that because we need to calculate the difference between prediction and actual y for ALL data points and take the average for each update, the x and y we feed in here is the entire dataset. 

5. **What does Epoch mean**: for each epoch we calculate the loss for the entire dataset, that gives us the loss for one gradient update. We need to iterate through the dataset for a few times (several epochs) for gradient descent. 

6. **What library we use for layer normalization**: we use standard scaler to conver x into mean = 0 std = 1. Here we standard scale across each data sample, this is called layer norm. Batch norm, on the other hand, is to standard scale across each feature (all data samples of that feature), and thus require to remember the population mean and std. 

7. **What should we plot to find if we have choosen the learning rate properly**: plot loss at each epoch against the number of epochs. Not plotting the parameter values

8. **Try out different types of learning rate**: constant, learning rate decay (1/k), momentum. 

## w_init
From w_init.ipynb, the only thing that needs to pay attention is the w_init dimension. 

## keras_first_experience
From keras_first_experience.ipynb, we test out how to construct a keras model for binary classification/multi-class classification/regression, and how to preprocess data for each scenario (one hot encoding and label encoding).

There is also an experiment for k-fold. 
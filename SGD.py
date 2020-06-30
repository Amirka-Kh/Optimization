import taichi as ti
import numpy as np
import tensorflow
from tensorboardX import SummaryWriter

#import mnist
from tensorflow.keras.datasets import mnist

# Initialization
ti.init(arch=ti.gpu)

# Data type shortcuts
real = ti.f32
scalar = lambda: ti.var(dt=real)

# Number of digits to recognise
n_numbers = 10

# Image size
n_pixels = 28 ** 2
image_size = 28

# Hiden nodes
n_hidden = 500

# Training epochs
training_epochs = 5

# Data types
pixels = scalar()
weights1 = scalar()
output1 = scalar()
weights2 = scalar()
output2 = scalar()
output2_exp = scalar()
output2_norm = scalar()
needed = scalar()
output_sum = scalar()
loss = scalar()
learning_rate = scalar()

# Data layout configuration, for fast computation
@ti.layout
def place():
    ti.root.dense(ti.i, n_pixels).place(pixels)
    ti.root.dense(ti.ij, (n_pixels, n_hidden)).place(weights1)
    ti.root.dense(ti.i, n_hidden).place(output1)
    ti.root.dense(ti.ij, (n_hidden, n_numbers)).place(weights2)
    ti.root.dense(ti.i, n_numbers).place(output2)
    ti.root.dense(ti.i, n_numbers).place(output2_exp)
    ti.root.dense(ti.i, n_numbers).place(output2_norm)
    ti.root.dense(ti.i, n_numbers).place(needed)
    ti.root.place(output_sum)
    ti.root.place(loss)
    ti.root.place(learning_rate)

    # Add gradient variables
    ti.root.lazy_grad()

# Initialize network
@ti.kernel
def init_weights_biases():
    # Layer 1
    for i in range(n_pixels):
        for j in range(n_hidden):
            weights1[i, j] = ti.random() * 0.005

    # Layer 2
    for i in range(n_hidden):
        for j in range(n_numbers):
            weights2[i, j] = ti.random() * 0.005

# Clear gradients and outputs
@ti.kernel
def clear_weights_biases_grad():
    # Layer 1
    for i in range(n_pixels):
        for j in range(n_hidden):
            weights1.grad[i, j] = 0

    # Layer 2
    for i in range(n_hidden):
        for j in range(n_numbers):
            weights2.grad[i, j] = 0

def clear_outputs_grad():
    # Layer 1
    for i in range(n_hidden):
        output1[i] = 0
        output1.grad[i] = 0
    # Layer 2
    for i in range(n_numbers):
        output2[i] = 0
        output2.grad[i] = 0
        output2_exp[i] = 0
        output2_exp.grad[i] = 0
        output2_norm[i] = 0
        output2_norm.grad[i] = 0

# Compute layers
@ti.kernel
def layer1():
    for i in range(n_pixels):
        for j in range(n_hidden):
            output1[j] += pixels[i] * weights1[i, j]

    for i in range(n_hidden):
        output1[i] = ti.tanh(output1[i])

@ti.kernel
def layer2():
    for i in range(n_hidden):
        for j in range(n_numbers):
            output2[j] += output1[i] * weights2[i, j]

    for i in range(n_numbers):
        output2_exp[i] = ti.exp(output2[i])
        output_sum[None] += output2_exp[i] + 1e-6

        # Normalization
    for i in range(n_numbers):
        output2_norm[i] = output2_exp[i] / output_sum[None]

# Compute loss (cross-entropy)
@ti.kernel
def compute_loss():
    for i in range(n_numbers):
        loss[None] += (-needed[i]) * ti.log(output2_norm[i])

# Gradient descent
@ti.kernel
def gd_layer1():
    for i in range(n_pixels):
        for j in range(n_hidden):
            weights1[i, j] -= learning_rate * weights1.grad[i, j]

@ti.kernel
def gd_layer2():
    for i in range(n_hidden):
        for j in range(n_numbers):
            weights2[i, j] -= learning_rate * weights2.grad[i, j]

# Step forward through network
def forward():
    layer1()
    layer2()
    compute_loss()

# Step back to coumpute gradients
def backward_grad():
    compute_loss.grad()
    layer2.grad()
    layer1.grad()

# MNIST images
(train_images, Y_train), (test_images, test_label) = mnist.load_data()

train_label = []
for y in Y_train:
    e = np.zeros((10, 1))
    e[y] = 1.0
    train_label.append(e)

# Initialize network
init_weights_biases()

# Compute accuracy of predictions on tests
def test_accuracy():
    n_test = len(test_images) // 10
    accuracy = 0
    for i in range(n_test):
        # Input
        curr_image = test_images[i]
        for j in range(image_size):
            for k in range(image_size):
                pixels[image_size * j + k] = float(curr_image[j][k])
        for j in range(n_numbers):
            needed[j] = int(test_label[i] == j)

        clear_outputs_grad()
        clear_weights_biases_grad()
        loss[None] = 0

        forward()
        writer = SummaryWriter("summaries ")
        writer.add_scalar("loss", loss[None], i)

        outputs = []
        for j in range(n_numbers):
            outputs.append(output2[j])
        # Digit with high prediction
        prediction = outputs.index(max(outputs))
        accuracy += int(prediction == test_label[i])

    return accuracy / n_test

# Training
def main():
    losses = []
    accuracies = []
    for n in range(training_epochs):
        #mini_bathes = [train_images[30*k:30*k+ 30] for k in range(0, 60000, 30)]
        #for mini_bath in mini_bathes:

        for i in range(len(train_images)):
            learning_rate[None] = 5e-3 *  (0.1 ** (2 * i // 60000))

            for j in range(image_size):
                for k in range(image_size):
                    pixels[image_size * j + k] = float(train_images[i][j][k])
            for j in range(n_numbers):
                needed[j] = int(train_label[i][j] == j)

            clear_outputs_grad()
            clear_weights_biases_grad()
            output_sum[None] = 0
            loss[None] = 0

            forward()

            curr_loss = loss[None]
            losses.append(curr_loss)
            losses = losses[-100:]
            if i % 100 == 0:
                print('i =', i, ' loss : ', sum(losses) / len(losses))
            if i % 1000 == 0:
                curr_acc = test_accuracy()
                print('test_accuracy: {:.2f}%'.format(100 * curr_acc))
                accuracies.append(curr_acc)

            loss.grad[None] = 1
            output_sum.grad[None] = 0

            backward_grad()

            gd_layer1()
            gd_layer2()


if __name__ == '__main__':
    main()

import taichi
import numpy as np
#import mnist
from tensorflow.keras.datasets import mnist

# Initialization
ti.init(arch=ti.gpu, defaul_fp=ti.f32)

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

# Betas
beta1 = 0.9
beta2 = 0.99

# Training epochs
training_epochs = 5

# Number of Momos
moment_to_hidden = n_pixels*n_hidden
moment_to_final = n_hidden*n_numbers

# Data types
pixels = scalar()
m_1 = scalar()
v_1 = scalar()
m_2 = scalar()
v_2 = scalar()
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
N_sma_max = scalar()
N_sma = scalar()
power = scalar()
rect_term = scalar()


# Data layout configuration, for fast computation
@ti.layout
def place():
    ti.root.dense(ti.i, moment_to_hidden).place(m_1)
    ti.root.dense(ti.i, moment_to_final).place(m_2)
    ti.root.dense(ti.i, moment_to_hidden).place(v_1)
    ti.root.dense(ti.i, moment_to_final).place(v_2)
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
    ti.root.place(N_sma_max)
    ti.root.place(N_sma)
    ti.root.place(power)
    ti.root.place(rect_term)

    # Add gradient variables
    ti.root.lazy_grad()


# Initialize network
@ti.kernel
def init_weights_biases():
    # Layer 1
    for i in range(n_pixels):
        for j in range(n_hidden):
            m_1[j] = 0
            v_1[j] = 0
            weights1[i, j] = ti.random() * 0.005

    # Layer 2
    for i in range(n_hidden):
        for j in range(n_numbers):
            m_2[j] = 0
            v_2[j] = 0
            weights2[i, j] = ti.random() * 0.005


# Clear gradients and outputs
@ti.kernel
def clear_all():
    # Layer 1
    for i in range(n_pixels):
        for j in range(n_hidden):
            output1[j] = 0
            output1.grad[j] = 0
            weights1.grad[i, j] = 0
            m_1[j] = 0
            v_1[j] = 0

    # Layer 2
    for i in range(n_hidden):
        for j in range(n_numbers):
            weights2.grad[i, j] = 0
            m_2[j] = 0
            v_2[j] = 0
            output2[j] = 0
            output2.grad[j] = 0
            output2_exp[j] = 0
            output2_exp.grad[j] = 0
            output2_norm[j] = 0
            output2_norm.grad[j] = 0


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


# Gradient descent, RAdam approach
@ti.kernel
def gd_layer1():
    for i in range(n_pixels):
        for j in range(n_hidden):
            index = n_hidden*i + j
            g = weights1.grad[i, j]
            m_1[index] = beta1 * m_1[index] + (1 - beta1) * g
            v_1[index] = beta2 * v_1[index] + (1 - beta2) * g * g
            m_hat = m_1[index] / (1 - pow(beta1, power[None]))
            dthetha1 = learning_rate * m_hat
            weights1[i, j] -= dthetha1


@ti.kernel
def gd_layer2():
    for i in range(n_hidden):
        for j in range(n_numbers):
            g = weights2.grad[i, j]
            index = n_hidden*i + j
            m_2[index] = beta1 * m_2[index] + (1 - beta1) * g
            v_2[index] = beta2 * v_2[index] + (1 - beta2) * g * g
            m_hat = m_2[index] / (1 - pow(beta1, power[None]))
            dthetha2 = learning_rate * m_hat
            weights2[i, j] -= dthetha2

@ti.kernel
def gdz_layer1():
    for i in range(n_pixels):
        for j in range(n_hidden):
            index = n_hidden*i + j
            g = weights1.grad[i, j]
            m_1[index] = beta1 * m_1[index] + (1 - beta1) * g
            v_1[index] = beta2 * v_1[index] + (1 - beta2) * g * g
            m_hat = m_1[index] / (1 - pow(beta1, power[None]))
            v_hat = np.sqrt(v_1[index] / (1 - pow(beta2, power[None])))
            dthetha1 = learning_rate / v_hat * m_hat * rect_term[None]
            weights1[i, j] -= dthetha1

@ti.kernel
def gdz_layer2():
    for i in range(n_hidden):
        for j in range(n_numbers):
            g = weights2.grad[i, j]
            index = n_hidden*i + j
            m_2[index] = beta1 * m_2[index] + (1 - beta1) * g
            v_2[index] = beta2 * v_2[index] + (1 - beta2) * g * g
            m_hat = m_2[index] / (1 - pow(beta1, power[None]))
            v_hat = np.sqrt(v_2[index] / (1 - pow(beta2, power[None])))
            dthetha2 = learning_rate / v_hat * m_hat * rect_term[None]
            weights2[i, j] -= dthetha2
   


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

        clear_all()
        loss[None] = 0

        forword()

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
    decay[None] = 0
    learning_rate[None] = 0.001
    weight_decay[None] = 0.0
    initial_decay = 0
    for n in range(training_epochs):
        for i in range(len(train_images)):
            if initial_decay > 0:
                learning_rate[None] = learning_rate[None]*(1. / (1. + decay[None]*float(i))

            for j in range(image_size):
                for k in range(image_size):
                    pixels[image_size * j + k] = float(train_images[i][j][k])
            for j in range(n_numbers):
                needed[j] = int(train_label[i][j] == j)

            clear_outputs_grad()
            clear_weights_biases_grad()
            output_sum[None] = 0
            loss[None] = 0
            power[None] = i

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
            
            beta2_i = beta2 ** i
            N_sma_max[None] = 2 / (1 - beta2) - 1 
            N_sma[None] = N_sma_max[None] - 2 * i * beta2_i / (1 - beta2_i)
            
            N_m = N_sma
            N_max = N_sma_max
            
            if i % 5 == 0:
                rect_term[None] = math.sqrt((N_m - 4) / N_max - 4) * (N_m - 2) / N_m * N_max / (N_max - 2)) 
                gdz_layer1()
                gdz_layer2()
            else:
                gd_layer1()
                gd_layer2()
                
                


if __name__ == '__main__':
    main()

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)  # suppress warnings


def create_dataset(shape, w_star, x_range, sample_size, sigma, seed=None):
    """
    Create linear regression dataset
    :param w_star: polynomial coefficients
    :param x_range: interval in which x should be
    :param sample_size: the number of elements in the dataset
    :param sigma: standard deviation of the distribution
    :param seed: random seed used to initialize the pseudo-random number generator
    :return: X, y that are two np.arrays containing the coordinates of the points of the dataset
    """
    random_state = np.random.RandomState(seed)
    x = random_state.uniform(x_range[0], x_range[1], sample_size)
    X_three = extract_feature_map(x, w_star.shape[0] - 1)
    X = extract_feature_map(x, shape - 1)

    y = X_three.dot(w_star)

    if sigma > 0:
        y += random_state.normal(0.0, sigma, sample_size)

    return X, y


def extract_feature_map(x, degree=3):
    """
    :param x: base number raised to the first "degree" powers
    :param degree: exponents
    :return f: a matrix containing the first 'degree' powers of 'x'
    """
    f = np.zeros([x.shape[0], degree + 1])

    for d in range(degree + 1):
        f[:, d] = x ** d

    return f


def extract_y(x, coefficients, degree):
    """
    :param x: a matrix containing in each row the first 'degree' powers of a random number in the interval [-3, 2]
    :param coefficients: vector of coefficients w_star' (in ascending order: from x**0 to x**n)
    :return y : value of y that satisfy the polynomial given 'x' and 'coefficients'
    """
    y = np.matmul(extract_feature_map(x, degree), coefficients)

    return y


def net_vars(n_dimensions, learning_rate):
    """
    Define the variables used in the current model
    :param n_dimensions:
    :param learning_rate:
    :return initializer, summaries, X, y, w, loss, train:
    """

    with tf.variable_scope("model_{}".format(n_dimensions)):
        # Placeholder for the data matrix, where each observation is a row
        X = tf.placeholder(tf.float32, shape=(None, n_dimensions), name='xx')
        y = tf.placeholder(tf.float32, shape=(None,), name='yy')

        # Variable for the model parameters
        w = tf.Variable(tf.zeros((n_dimensions, 1)), trainable=True, name='weights')

        # Loss function
        prediction = tf.reshape(tf.matmul(X, w), (-1,))
        loss = tf.reduce_mean(tf.square(y - prediction))

        s_loss = tf.summary.scalar('loss', loss)

        # Gradient descent update operation
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

        initializer = tf.variables_initializer([w])

        # Merges all summaries into single a operation
        summaries = tf.summary.merge([s_loss])

    return initializer, summaries, X, y, w, loss, train


def main(net_vars, n_iterations, sample_size, sigma, w_star, x_range, plot_directory, degrees=None,
         early_stopping=False):
    """
    :param net_vars: initializer, summaries, X, y, w, loss, train
    :param n_iterations: number of iterations of the algorithm
    :param sample_size: vector containing train and validation number of samples
    :param sigma: standard deviation of the distribution (for generating the dataset)
    :param w_star: coefficients of the given polynomial
    :param x_range: interval in which x should be
    :param degrees: degree of the true and the estimated polynomial
    """

    initializer, summaries, X, y, w, loss, train = net_vars

    # Create and plot the datasets
    X_train, y_train = create_dataset(X.shape[1], w_star, x_range, sample_size[0], sigma, 0)
    X_val, y_val = create_dataset(X.shape[1], w_star, x_range, sample_size[1], sigma, 1)

    plt.plot(X_train[:, 1], y_train, '.', label='Train')
    plt.plot(X_val[:, 1], y_val, '.', label='Validation')

    plt.xlabel('x', fontsize=11)
    plt.ylabel('y', fontsize=11)
    plt.legend(fontsize=10, fancybox=True)
    plt.title('Dataset', weight='bold', fontsize=12)

    plt.savefig(plot_directory + '-dataset', )
    plt.show()

    with tf.Session() as session:
        # Creating object that writes graph structure and summaries to disk
        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'tmp/train' + plot_directory.replace('img/', '')
        test_log_dir = 'tmp/validation' + plot_directory.replace('img/', '')
        writer_train = tf.summary.FileWriter(train_log_dir, session.graph)
        writer_validation = tf.summary.FileWriter(test_log_dir, session.graph)

        session.run(initializer)

        validation_loss = 0
        if early_stopping:
            best_validation_loss = np.inf
            best_iteration = -1

        for t in range(1, n_iterations + 1):
            s, train_loss, _ = session.run([summaries, loss, train], feed_dict={X: X_train, y: y_train})
            print('Iteration {0}. Loss: {1}.'.format(t, train_loss))
            f.write('Iteration {0}. Loss: {1}.\n'.format(t, train_loss))

            # Stores the summaries for iteration t
            writer_train.add_summary(s, t)

            s, validation_loss = session.run([summaries, loss], feed_dict={X: X_val, y: y_val})
            writer_validation.add_summary(s, t)

            if early_stopping:
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_iteration = t
                else:
                    if t - best_iteration > 10:
                        break

        if early_stopping:
            f.write("The best loss of %.5f is reached after %d iterations.\n" % (
                best_validation_loss, best_iteration))

        print('Validation loss: {0}.'.format(validation_loss))
        f.write('Validation loss: {0}.\n'.format(validation_loss))

        weights = session.run(w).reshape(-1)
        print(weights)
        f.write('weights: ' + str(weights))

        writer_train.close()
        writer_validation.close()

        # Find w_hat, that is an estimate of w_star, and plot the original polynomial and the estimated one
        w_hat = session.run(w)

        x = np.linspace(x_range[0], x_range[1])

        plt.plot(x, extract_y(x, w_star, 3), label='Polynomial')

        if degrees is None:
            plt.plot(x, extract_y(x, w_hat, 3), label='Estimated polynomial (3rd degree)')
        else:
            plt.plot(x, extract_y(x, w_hat[:-1], 3), label='Estimated polynomial (3rd degree)')
            plt.plot(x, extract_y(x, w_hat, 4), label='Estimated polynomial (4th degree)')

        plt.xlabel('x', fontsize=11)
        plt.ylabel('y', fontsize=11)

        plt.legend(fontsize=10, fancybox=True)
        plt.title('Polynomial', weight='bold', fontsize=12)

        plt.savefig(plot_directory + '-polynomial')
        plt.show()

        # Plot dataset and estimation
        plt.plot(X_train[:, 1], y_train, '.', label='Train dataset')
        plt.plot(X_val[:, 1], y_val, '.', label='Validation dataset')

        plt.plot(x, extract_y(x, w_star, 3), label='Polynomial')

        if degrees is None:
            plt.plot(x, extract_y(x, w_hat, 3), label='Estimated polynomial (3rd degree)')
        else:
            plt.plot(x, extract_y(x, w_hat[:-1], 3), label='Estimated polynomial (3rd degree)')
            plt.plot(x, extract_y(x, w_hat, 4), label='Estimated polynomial (4th degree)')

        plt.xlabel('x', fontsize=11)
        plt.ylabel('y', fontsize=11)
        plt.legend(fontsize=10, fancybox=True)
        # plt.title('Polynomial', weight='bold', fontsize=12)

        plt.savefig(plot_directory + '-polynomial+dataset')
        plt.show()

        session.close()


################################################################################


n_iterations = 2000
sample_size = [100, 100]  # [train, validation]
n_dimensions = 4
sigma = 0.5

# Naming constants/variables to facilitate inspection
learning_rate = tf.constant(1e-2, dtype=tf.float32, name='learning_rate')
net_vars1 = net_vars(n_dimensions, learning_rate)

w_star = np.array([-8, -4, 2, 1], dtype=np.float32)
x_range = [-3, 2]

plot_directory = "img/"
if not os.path.isdir(plot_directory):
    os.mkdir(plot_directory)

out_directory = "out/"
if not os.path.isdir(out_directory):
    os.mkdir(out_directory)

# Run the script with 3000 iterations with the early stopping for choosing the number of iterations
f = open(out_directory + "model1-es.txt", "w")
main(net_vars1, n_iterations, sample_size, sigma, w_star, x_range, plot_directory + 'model1-es', early_stopping=True)
f.close()

# Change the number of iterations
n_iterations = 1248

f = open(out_directory + "model1.txt", "w")
main(net_vars1, n_iterations, sample_size, sigma, w_star, x_range, plot_directory + 'model1', early_stopping=True)
f.close()

### Change the training dataset size to 50, 10, and 5 observations

f = open(out_directory + "model2.txt", "w")
sample_size = [50, 100]  # [train, validation]
main(net_vars1, n_iterations, sample_size, sigma, w_star, x_range, plot_directory + 'model2')
f.close()

f = open(out_directory + "model3.txt", "w")
sample_size = [10, 100]  # [train, validation]
main(net_vars1, n_iterations, sample_size, sigma, w_star, x_range, plot_directory + 'model3')
f.close()

f = open(out_directory + "model4.txt", "w")
sample_size = [5, 100]  # [train, validation]
main(net_vars1, n_iterations, sample_size, sigma, w_star, x_range, plot_directory + 'model4')
f.close()

### Document what happens when σ is increased to 2, 4, and 8
sample_size = [100, 100]  # reset sample size

f = open(out_directory + "model5.txt", "w")
sigma = 2
main(net_vars1, n_iterations, sample_size, sigma, w_star, x_range, plot_directory + 'model5')
f.close()

f = open(out_directory + "model6.txt", "w")
sigma = 4
main(net_vars1, n_iterations, sample_size, sigma, w_star, x_range, plot_directory + 'model6')
f.close()

f = open(out_directory + "model7.txt", "w")
sigma = 8
main(net_vars1, n_iterations, sample_size, sigma, w_star, x_range, plot_directory + 'model7')
f.close()

### Reduce your training dataset to 10 observations, and compare fitting a polynomial of degree three (as before)
# with fitting a polynomial of degree four (which does not match the underlying polynomial p).
# Plot the resulting polynomials and document the validation loss.
f = open(out_directory + "model8.txt", "w")
sample_size = [10, 100]  # [train, validation]
sigma = 0.5
degrees = [3, 4]

n_dimensions = 5
net_vars1 = net_vars(n_dimensions, learning_rate)

main(net_vars1, n_iterations, sample_size, sigma, w_star, x_range, plot_directory + 'model8', degrees)
f.close()

# Test with more iterations
f = open(out_directory + "model8-test.txt", "w")
n_iterations = 8000
main(net_vars1, n_iterations, sample_size, sigma, w_star, x_range, plot_directory + 'model8-test', degrees)
f.close()

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)  # suppress warnings

# More specifically, find an estimate of w∗ = [−8, −4, 2, 1]^T supposing that such vector is unknown.
# Each xi should be in the interval [−3, 2].
# Use a sample of size 100 created with a seed of 0 for training, and a sample of size 100 created with a seed of 1
# for validation. Let σ = 1/2.


def create_dataset(w_star, x_range, sample_size, sigma, seed=None):
    """Create linear regression dataset (without bias term)"""

    random_state = np.random.RandomState(seed)
    x = random_state.uniform(x_range[0], x_range[1], sample_size)
    X = np.zeros((sample_size, w_star.shape[0]))
    for i in range(sample_size):
        X[i, 0] = 1.
        for j in range(1, w_star.shape[0]):
            X[i, j] = x[i] ** j

    y = X.dot(w_star)

    if sigma > 0:
        y += random_state.normal (0.0, sigma, sample_size )

    return X, y


def main(n_iterations, sample_size, n_dimensions, sigma, learning_rate, w_star, x_range):
    # Placeholder for the data matrix, where each observation is a row
    X = tf.placeholder(tf.float32, shape=(None, n_dimensions), name='X')  # Placeholder for the targets
    y = tf.placeholder(tf.float32, shape=(None,), name='y')

    # Variable for the model parameters
    w = tf.Variable(tf.zeros((n_dimensions, 1)), trainable=True, name='w')

    # Loss function
    prediction = tf.reshape(tf.matmul(X, w), (-1,))
    loss = tf.reduce_mean(tf.square(y - prediction))

    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    train = optimizer.minimize(loss)  # Gradient descent update operation

    initializer = tf.global_variables_initializer()

    X_train, y_train = create_dataset(w_star, x_range, sample_size[0], sigma, 0)
    X_val, y_val = create_dataset(w_star, x_range, sample_size[1], sigma, 1)

    # Plot dataset
    plt.plot(X_train[:, 1], y_train, '.')
    plt.plot(X_val[:, 1], y_val, '.')

    plt.show()

    # Merges all summaries into single a operation
    summaries = tf.summary.merge_all()

    session = tf.Session()

    # Creating object that writes graph structure and summaries to disk
    writer_train = tf.summary.FileWriter('tmp/train')
    writer_validation = tf.summary.FileWriter('tmp/validation')

    session.run(initializer)

    validation_loss = 0
    for t in range(1, n_iterations + 1):
        s, train_loss, _ = session.run([summaries, loss, train], feed_dict={X: X_train, y: y_train})
        print('Iteration {0}. Loss: {1}.'.format(t, train_loss))

        # Stores the summaries for iteration t
        writer_train.add_summary(s, t)

        s, validation_loss = session.run([summaries, loss], feed_dict={X: X_val, y: y_val})
        writer_validation.add_summary(s, t)

    print('Validation loss: {0}.'.format(validation_loss))
    print(session.run(w).reshape(-1))

    writer_train.close()
    writer_validation.close()

    # Plot polynom
    w_hat = session.run(w)
    x = np.linspace(x_range[0], x_range[1])

    plt.plot(x, extract_y(x, w_star))
    plt.plot(x, extract_y(x, w_hat))

    plt.show()

    session.close()


def extract_feature_map(x, degree=3):
    f = np.zeros([x.shape[0], degree + 1])

    for d in range(degree + 1):
        f[:, d] = x ** d
    return f


def extract_y(x, coefficients):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    y = np.matmul(extract_feature_map(x, 3), coefficients)
    return y


################################################################################


n_iterations = 1100
sample_size = [100, 100]  # [train, validatioon]
n_dimensions = 4
sigma = 0.5

# Naming constants/variables to facilitate inspection
learning_rate = tf.constant(1e-2, dtype=tf.float32, name='learning_rate')

w_star = np.array([-8, -4, 2, 1], dtype=np.float32)
x_range = [-3, 2]

# run the script
main(n_iterations, sample_size, n_dimensions, sigma, learning_rate, w_star, x_range)

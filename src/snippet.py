import numpy as np
import tensorflow as tf

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


def main():
    sample_size_train = 100

    sample_size_val = 100
    n_dimensions = 10
    sigma = 0.5
    n_iterations = 20
    learning_rate = 0.5

    w_star = np.array([-8, -4, 2, 1], dtype=np.float32)
    x_range = [-3, 2]

    # Placeholder for the data matrix, where each observation is a row
    X = tf.placeholder(tf.float32, shape=(None, n_dimensions))  # Placeholder for the targets
    y = tf.placeholder(tf.float32, shape=(None,))

    # Variable for the model parameters
    w = tf.Variable(tf.zeros((n_dimensions, 1)), trainable=True)

    # Loss function
    prediction = tf.reshape(tf.matmul(X, w), (-1,))
    loss = tf.reduce_mean(tf.square(y - prediction))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    train = optimizer.minimize(loss)  # Gradient descent update operation

    initializer = tf.global_variables_initializer()

    X_train, y_train = create_dataset(w_star, x_range, sample_size_train, sigma, 0)

    session = tf.Session()
    session.run(initializer)

    for t in range(1, n_iterations + 1):
        l, _ = session.run([loss, train], feed_dict={X: X_train, y: y_train})
        print('Iteration {0}. Loss: {1}.'.format(t, l))

    X_val, y_val = create_dataset(w_star, x_range, sample_size_val, sigma, 1)

    l = session.run(loss, feed_dict={X: X_val, y: y_val})

    print('Validation loss: {0}.'.format(l))
    print(session.run(w).reshape(-1))

    session.close()


tf.logging.set_verbosity(tf.logging.ERROR)  # suppress warnings
main()  # run the script

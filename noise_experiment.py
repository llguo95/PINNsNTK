from main import main
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf

## LOSS OPTIMIZER STUDY
starter_learning_rate = 1e-5
global_step = tf.Variable(0, trainable=False)
learning_rate_scheme = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.9, staircase=False)

train_algo = tf.train.GradientDescentOptimizer(learning_rate=starter_learning_rate)

noise_bools = [False, True]

for i, noise_bool in enumerate(noise_bools):
    main(train_algo=train_algo, visualize=["eigenvalues", "prediction", "convergence"], SMOKE_TEST=True, iteration=i, noisy_data=noise_bool)

# print('r2 score:', r2_score(u_star, u_pred))

plt.figure(num="eigenvalues")
plt.legend(['Noiseless', 'Noise'])
plt.savefig('noise_eigenvalues.png')

fig = plt.figure(num="prediction")
fig.get_axes()[0].legend(['Exact', 'Noiseless', 'Noise'])
fig.get_axes()[1].legend(['Noiseless', 'Noise'])

plt.figure(num="convergence")
plt.legend(['Noiseless', 'Noise'])

plt.show()

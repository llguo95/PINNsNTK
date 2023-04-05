from main import main
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf

## LOSS OPTIMIZER STUDY
starter_learning_rate = 1e-5
global_step = tf.Variable(0, trainable=False)
learning_rate_scheme = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.9, staircase=False)

train_algos = [
    tf.train.GradientDescentOptimizer(learning_rate=starter_learning_rate),
    tf.train.AdagradOptimizer(learning_rate=starter_learning_rate),
    # tf.train.MomentumOptimizer(learning_rate=starter_learning_rate, momentum=1e-3),
    # tf.train.AdamOptimizer(learning_rate=starter_learning_rate),
    ]

for i, train_algo in enumerate(train_algos):
    main(train_algo=train_algo, visualize=["eigenvalues", "prediction", "convergence"], SMOKE_TEST=True, iteration=i,)

# print('r2 score:', r2_score(u_star, u_pred))
plt.figure(num="eigenvalues")
plt.legend(['GD', 'Adagrad', 'Momentum', 'Adam'])
plt.savefig('optimizers_eigenvalues.png')

fig = plt.figure(num="prediction")
fig.get_axes()[0].legend(['Exact', 'GD', 'Adagrad', 'Momentum', 'Adam'])
fig.get_axes()[1].legend(['GD', 'Adagrad', 'Momentum', 'Adam'])

plt.figure(num="convergence")
plt.legend(['GD', 'Adagrad', 'Momentum', 'Adam'])


plt.show()

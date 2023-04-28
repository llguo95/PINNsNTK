from main import main
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf
import pandas as pd

## LOSS OPTIMIZER STUDY (ADAPTIVE)
starter_learning_rate = 1e-5
global_step = tf.Variable(0, trainable=False)
learning_rate_scheme = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.9, staircase=False)

train_algos = [
    tf.train.GradientDescentOptimizer(learning_rate=learning_rate_scheme),
    tf.train.AdagradOptimizer(learning_rate=learning_rate_scheme),
    tf.train.MomentumOptimizer(learning_rate=learning_rate_scheme, momentum=1e-3),
    tf.train.AdamOptimizer(learning_rate=learning_rate_scheme),
    ]

df_errors = pd.DataFrame(index=['error_u', 'error_r'])

for i, train_algo in enumerate(train_algos):
    main(
        train_algo=train_algo, 
        visualize=["eigenvalues", "prediction", "convergence"], 
        SMOKE_TEST=False, 
        iteration=i, 
        noisy_data=False,
        df_errors=df_errors,
        )

df_errors.to_csv('data/optimizers_ada_prediction_errors.csv')

# print('r2 score:', r2_score(u_star, u_pred))
plt.figure(num="eigenvalues")
plt.legend(['GD', 'Adagrad', 'Momentum', 'Adam'])
plt.tight_layout()
plt.savefig('img/optimizers_ada_eigenvalues.png')

fig = plt.figure(num="prediction")
fig.get_axes()[0].legend(['Exact', 'GD', 'Adagrad', 'Momentum', 'Adam'])
fig.get_axes()[1].legend(['GD', 'Adagrad', 'Momentum', 'Adam'])
plt.tight_layout()
plt.savefig('img/optimizers_ada_prediction.png')

plt.figure(num="convergence")
plt.legend(['GD', 'Adagrad', 'Momentum', 'Adam'])
plt.tight_layout()
plt.savefig('img/optimizers_ada_convergence.png')


plt.show()

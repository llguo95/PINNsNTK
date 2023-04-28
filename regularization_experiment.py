from main import main
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf
import pandas as pd

## LOSS REGULARIZER STUDY
starter_learning_rate = 1e-5
global_step = tf.Variable(0, trainable=False)
learning_rate_scheme = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.9, staircase=False)

train_algo = tf.train.GradientDescentOptimizer(learning_rate=starter_learning_rate)

regularization_bools = [False, True]

df_errors = pd.DataFrame(index=['error_u', 'error_r'])

for i, regularization_bool in enumerate(regularization_bools):
    main(
        train_algo=train_algo, 
        visualize=["eigenvalues", "prediction", "convergence"], 
        SMOKE_TEST=False, 
        iteration=i, 
        noisy_data=False,
        regularization=regularization_bool,
        df_errors=df_errors,
        )

df_errors.to_csv('data/regularization_prediction_errors.csv')
# print('r2 score:', r2_score(u_star, u_pred))

plt.figure(num="eigenvalues")
plt.legend(['No regularization', 'Regularization'])
plt.tight_layout()
plt.savefig('img/regularization_eigenvalues.png')

fig = plt.figure(num="prediction")
fig.get_axes()[0].legend(['Exact', 'No regularization', 'Regularization'])
fig.get_axes()[1].legend(['No regularization', 'Regularization'])
plt.tight_layout()
plt.savefig('img/regularization_prediction.png')

plt.figure(num="convergence")
plt.legend(['No regularization', 'Regularization'])
plt.tight_layout()
plt.savefig('img/regularization_convergence.png')

plt.show()

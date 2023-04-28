from main import main
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf
import pandas as pd

## NOISE STUDY
starter_learning_rate = 1e-5
global_step = tf.Variable(0, trainable=False)
learning_rate_scheme = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.9, staircase=False)


train_algo = tf.train.GradientDescentOptimizer(learning_rate=starter_learning_rate)

noise_bools = [False, True]

for seed in range(1):
    df = pd.DataFrame()

    for i, noise_bool in enumerate(noise_bools):
        df = main(
            train_algo=train_algo, 
            visualize=["eigenvalues", "prediction", "convergence"], 
            SMOKE_TEST=True, 
            iteration=i, 
            noisy_data=noise_bool,
            df=df,
            seed=seed,
            )

    df.to_csv('data/noise_convergence_%d.csv' % seed)
    print(df)

    # print('r2 score:', r2_score(u_star, u_pred))

    plt.figure(num="eigenvalues")
    plt.legend(['Noiseless', 'Noise'])
    plt.tight_layout()
    plt.savefig('img/noise_eigenvalues_%d.png' % seed)

    fig = plt.figure(num="prediction")
    fig.get_axes()[0].legend(['Exact', 'Noiseless', 'Noise'])
    fig.get_axes()[1].legend(['Noiseless', 'Noise'])
    plt.tight_layout()
    plt.savefig('img/noise_prediction_%d.png' % seed)

    plt.figure(num="convergence")
    plt.legend(['Noiseless', 'Noise'])
    plt.tight_layout()
    plt.savefig('img/noise_convergence_%d.png' % seed)

plt.show()

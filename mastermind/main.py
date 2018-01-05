from .mastermind_nn import MastermindNN
import tensorflow as tf
from .board import *


def main():
    with tf.Graph().as_default():
        nn = MastermindNN()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            train(session, nn, '/home/teinvdlugt/Documents/AA Studie/Programmeren 1'
                               '/Practica/Week 7/mastermind_ai/data/training_data_100K', 100, 100, 1000000)


def train(session, nn, training_data, batch_size, eval_steps, max_steps):
    # Get the training data
    inputs = np.load(training_data + '_inputs.npy')
    outputs = np.load(training_data + '_outputs.npy')
    num_data_points = len(inputs)

    train_step = 0
    total_loss = 0  # Sum of all losses within single eval period
    last_eval_stap = 0  # number of the last time an eval was done
    while train_step < max_steps and (train_step + 1) * batch_size < num_data_points:
        x = inputs[train_step * batch_size: (train_step + 1) * batch_size]
        y_ = outputs[train_step * batch_size: (train_step + 1) * batch_size]

        loss = nn.train_step(session, x, y_)
        total_loss += loss

        # Eval
        if train_step % eval_steps == 0 or train_step == max_steps - 1:
            print('At training step', train_step)
            if train_step == 0:
                avg_loss = loss
            else:
                avg_loss = total_loss / (train_step - last_eval_stap)
            nn.write_summary(session, x, y_, train_step, avg_loss)

            total_loss = 0
            last_eval_stap = train_step

        train_step += 1


if __name__ == '__main__':
    main()

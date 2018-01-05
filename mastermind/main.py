from .mastermind_nn import MastermindNN
import tensorflow as tf
from .board import *


def main():
    with tf.Graph().as_default():
        nn = MastermindNN()

        with tf.Session() as session:
            train(session, nn,
                  '/home/teinvdlugt/Documents/Computer Science/Machine Learning/'
                  '/Mastermind/checkpoints/',
                  '/home/teinvdlugt/Documents/Computer Science/Machine Learning/'
                  '/Mastermind/data/training_data_100K_inputs.npy',
                  '/home/teinvdlugt/Documents/Computer Science/Machine Learning/'
                  '/Mastermind/data/training_data_100K_outputs.npy',
                  100, 100, 100000, 100000)


def train(session, nn, checkpoint_dir, training_data_inputs, training_data_outputs,
          batch_size, eval_steps, save_steps, max_steps):
    # Restore the variables of the network
    nn.restore_or_initialize(session, checkpoint_dir)

    # Get the training data
    print('Getting training data...')
    inputs = np.load(training_data_inputs)
    outputs = np.load(training_data_outputs)
    num_data_points = len(inputs)

    # Global step tensor:
    increment_global_step_op = nn.global_step.assign_add(1)

    train_step = 0
    total_loss = 0  # Sum of all losses within single eval period
    last_eval_stap = 0  # number of the last time an eval was done
    while train_step < max_steps and (train_step + 1) * batch_size <= num_data_points:
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
            nn.write_summary(session, x, y_, nn.global_step.eval(session), avg_loss)

            total_loss = 0
            last_eval_stap = train_step

        # Save a checkpoint
        if (train_step % save_steps == 0 and train_step != 0) or train_step == max_steps - 1:
            nn.save(session, checkpoint_dir, nn.global_step.eval(session))

        # Increment train step
        train_step += 1
        session.run(increment_global_step_op)


if __name__ == '__main__':
    main()

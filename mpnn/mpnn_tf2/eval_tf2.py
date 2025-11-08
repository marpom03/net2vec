import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import datetime
import os, random
import graph_nn2_tf2 as graph_nn
from graph_nn2_tf2 import parser as gparser

random.seed(0)
np.random.seed(0)

gparser.add_argument('--nval', type=int, default=32)
gparser.add_argument('--checkpoint', type=int, default=None)
args = gparser.parse_args()
graph_nn.args = args

def make_set():
    ds = tf.data.TFRecordDataset([args.eval])     
    ds = ds.map(graph_nn.parse)                  
    it = tf.compat.v1.data.make_one_shot_iterator(ds)
    with tf.device("/cpu:0"):
        return graph_nn.transformation_func(it, args.batch_size)

def main():
    REUSE=None
    g = tf.Graph()
    with g.as_default():
        tf.compat.v1.set_random_seed(0)
        global_step = tf.compat.v1.train.get_or_create_global_step()

        batch, labels = make_set()   
        model = graph_nn.MessagePassing()
        predictions = model(batch, training=False)

        loss = tf.compat.v1.losses.mean_squared_error(labels, predictions)
        # saver = tf.compat.v1.train.Saver(list(model.variables) + [global_step])
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    with tf.compat.v1.Session(graph=g) as ses:
        ses.run(tf.compat.v1.local_variables_initializer())
        ses.run(tf.compat.v1.global_variables_initializer())

        ckpt = tf.train.latest_checkpoint(args.log_dir)
        if args.checkpoint is not None:
            ckpt = os.path.join(args.log_dir, f"model.ckpt-{args.checkpoint}")

        if ckpt is None or not tf.io.gfile.exists(ckpt + ".index"):
            raise FileNotFoundError(f"Checkpoint doesn't exist: {ckpt}")

        print("Loading checkpoint:", ckpt)
        saver.restore(ses, ckpt)

        label_py, predictions_py = [], []
        collected = 0
        for _ in range(args.nval):              
            try:
                y, yhat, step = ses.run([labels, predictions, global_step])
            except tf.errors.OutOfRangeError:
                break
            label_py.append(y)
            predictions_py.append(yhat)
            collected += 1

        if collected == 0:
            print("No samples 'w'", args.eval)
            return

        label_py = np.concatenate(label_py, axis=0).reshape(-1)
        predictions_py = np.concatenate(predictions_py, axis=0).reshape(-1)

        print(label_py.shape)
        print('{} step: {} mse: {} R**2: {} Pearson: {}'.format(
            str(datetime.datetime.now()),
            step,
            np.mean((label_py - predictions_py)**2),
            graph_nn.fitquality(label_py, predictions_py),
            np.corrcoef(label_py, predictions_py, rowvar=False)[0, 1]
        ), flush=True)

        plt.figure()
        plt.plot(label_py, predictions_py, '.')
        graph_nn.line_1(label_py, label_py)
        plt.grid('on')
        plt.xlabel('Label')
        plt.ylabel('Prediction')
        plt.title('Evaluation at step {}'.format(step))
        plt.savefig('eval.pdf')
        plt.close()

        plt.figure()
        plt.hist(label_py - predictions_py, 50)
        plt.savefig('rez_hist.pdf')
        plt.close()

if __name__ == '__main__':
    main()

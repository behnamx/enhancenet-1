#!/usr/bin/env python
import os

import numpy as np
import tensorflow as tf
from tqdm import trange

import vgg16
from config import get_config, print_usage
from nnlib import *
from utils.cifar10 import load_data

PER_CHANNEL_MEANS = np.array([0.47614917, 0.45001204, 0.40904046])




data_dir = "/Users/kwang/Downloads/cifar-10-batches-py"


class MyNetwork(object):
    """Network class """

    def __init__(self, x_shp, config):

        self.config = config

        # Get shape
        self.x_shp = x_shp

        # Build the network
        self._build_placeholder()
        self._build_model()
        self._build_loss()
        self._build_optim()
        self._build_eval()
        self._build_summary()
        self._build_writer()

    def _build_placeholder(self):
        """Build placeholders."""

        # Get shape for placeholder
        x_in_shp = [None] + list(self.x_shp[1:])
        imgsize = TODO

        # Create Placeholders for inputs
        # self.x_in = tf.placeholder(tf.float32, shape=x_in_shp)
        # self.y_in = tf.placeholder(tf.int64, shape=(None, ))


        self.x_in = tf.placeholder(tf.float32, [None, imgsize[0], imgsize[1], imgsize[2]])
        self.y_in = tf.placeholder(tf.float32, [None, 4*imgsize[0], 4*imgsize[1], imgsize[2]])




    def _build_model(self):
        """Build our network."""

        rblock = [resi, [[conv], [relu], [conv]]]
        ys_est = NN('generator',
                    [self.x_in,
                     [conv], [relu],
                     rblock, rblock, rblock, rblock, rblock,
                     rblock, rblock, rblock, rblock, rblock,
                     [upsample], [conv], [relu],
                     [upsample], [conv], [relu],
                     [conv], [relu],
                     [conv, 3]])
        ys_res = tf.image.resize_images(xs, [4*imgsize[0], 4*imgsize[1]],
                                        method=tf.image.ResizeMethod.BICUBIC)


        # TODO: Also use tf.summary.image to visualize some images
        

        self.y_est += ys_res + PER_CHANNEL_MEANS


        # For VGG -- Follw instructions at https://github.com/machrisaa/tensorflow-vgg

        # Get vgg16.py, and the npy weights (this is for the estimate)
        vgg = vgg16.Vgg16()
        self.y_est_resize = tf.image.resize_images(
            self.y_est, [224, 244],
            method=tf.image.ResizeMethod.BICUBIC)
        vgg.build(self.y_est_resize)

        # e.g. get conv5
        self.vgg_est = vgg.conv5_3

        # For the grount truth (HR image)
        TODO


    def _build_loss(self):
        """Build our cross entropy loss."""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):

            self.y_est
            self.y_in

            self.loss = TODO

            # Record summary for loss
            tf.summary.scalar("loss", self.loss)

    def _build_optim(self):
        """Build optimizer related ops and vars."""

        with tf.variable_scope("Optim", reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable(
                "global_step", shape=(),
                initializer=tf.zeros_initializer(),
                dtype=tf.int64,
                trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate)
            self.optim = optimizer.minimize(
                self.loss, global_step=self.global_step)

    def _build_eval(self):
        """Build the evaluation related ops"""

        with tf.variable_scope("Eval", tf.AUTO_REUSE):

            # Compute the accuracy of the model. When comparing labels
            # elemwise, use tf.equal instead of `==`. `==` will evaluate if
            # your Ops are identical Ops.

            # get PSNR

            self.acc = TODO, put PSNR equation here, using y_in and y_est

            # Record summary for accuracy
            tf.summary.scalar("psnr", self.acc)

            # DONE: We also want to save best validation accuracy. So we do
            # something similar to what we did before with n_mean
            self.best_va_acc_in = tf.placeholder(tf.float32, shape=())
            self.best_va_acc = tf.get_variable(
                "best_va_acc", shape=(), trainable=False)
            # Assign op to store this value to TF variable
            self.acc_assign_op = tf.assign(
                self.best_va_acc, self.best_va_acc_in)

    def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()

    def _build_writer(self):
        """Build the writers and savers"""

        # Create summary writers (one for train, one for validation)
        self.summary_tr = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, "train"))
        self.summary_va = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, "valid"))
        # Create savers (one for current, one for best)
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()
        # Save file for the current model
        self.save_file_cur = os.path.join(
            self.config.log_dir, "model")
        # Save file for the best model
        self.save_file_best = os.path.join(
            self.config.save_dir, "model")

    def train(self, x_tr, y_tr, x_va, y_va):
        """Training function.

        Parameters
        ----------
        x_tr : ndarray
            Training low resolution image

        y_tr : ndarray
            Training high resolution image

        x_va : ndarray
            Validation low resolution image

        y_va : ndarray
            Validation high resolution image

        """

        # ----------------------------------------
        # Run TensorFlow Session
        with tf.Session() as sess:
            # Init
            print("Initializing...")
            sess.run(tf.global_variables_initializer())

            # Assign normalization variables from statistics of the train data
            sess.run(self.n_assign_op, feed_dict={
                self.n_mean_in: x_tr_mean,
                self.n_range_in: x_tr_range,
            })

            # DONE: Check if previous train exists
            latest_checkpoint = tf.train.latest_checkpoint(
                self.config.log_dir)
            b_resume = tf.train.latest_checkpoint(
                self.config.log_dir) is not None
            if b_resume:
                # DONE: Restore network
                print("Restoring from {}...".format(
                    self.config.log_dir))
                self.saver_cur.restore(
                    sess,
                    latest_checkpoint
                )
                # DONE: restore number of steps so far
                step = sess.run(self.global_step)
                # DONE: restore best acc
                best_acc = sess.run(self.best_va_acc)
            else:
                print("Starting from scratch...")
                step = 0
                best_acc = 0

            print("Training...")
            batch_size = config.batch_size
            max_iter = config.max_iter
            # For each epoch
            for step in trange(step, max_iter):

                # Get a random training batch
                ind_cur = np.random.choice(
                    len(x_tr), batch_size, replace=False)
                x_b = np.array([x_tr[_i] for _i in ind_cur])
                y_b = np.array([y_tr[_i] for _i in ind_cur])

                # DONE: Write summary every N iterations as well as the first
                # iteration. Use `self.config.report_freq`. Make sure that we
                # write at the first iteration, and every kN iterations where k
                # is an interger. HINT: we write the summary after we do the
                # optimization.
                b_write_summary = ((step + 1) % self.config.report_freq) == 0
                b_write_summary = b_write_summary or step == 0
                if b_write_summary:
                    fetches = {
                        "optim": self.optim,
                        "summary": self.summary_op,
                        "global_step": self.global_step,
                    }
                else:
                    fetches = {
                        "optim": self.optim,
                    }

                # Run the operations necessary for training
                res = sess.run(
                    fetches=fetches,
                    feed_dict={
                        self.x_in: x_b,
                        self.y_in: y_b,
                    },
                )

                # Write Training Summary if we fetched it (don't write meta
                # graph). See that we actually don't need the above
                # `b_write_summary` actually :-)
                if "summary" in res:
                    self.summary_tr.add_summary(
                        res["summary"], global_step=res["global_step"],
                    )
                    self.summary_tr.flush()

                    # Also save current model to resume when we write the
                    # summary.
                    self.saver_cur.save(
                        sess, self.save_file_cur,
                        global_step=self.global_step,
                        write_meta_graph=False,
                    )

                # DONE: Validate every N iterations and at the first
                # iteration. Use `self.config.val_freq`. Make sure that we
                # validate at the correct iterations. HINT: should be similar
                # to above.
                b_validate = ((step + 1) % self.config.val_freq) == 0
                b_validate = b_validate or step == 0
                if b_validate:
                    res = sess.run(
                        fetches={
                            "acc": self.acc,
                            "summary": self.summary_op,
                            "global_step": self.global_step,
                        },
                        feed_dict={
                            self.x_in: x_va,
                            self.y_in: y_va
                        })
                    # Write Validation Summary
                    self.summary_va.add_summary(
                        res["summary"], global_step=res["global_step"],
                    )
                    self.summary_va.flush()

                    # If best validation accuracy, update W_best, b_best, and
                    # best accuracy. We will only return the best W and b
                    if res["acc"] > best_acc:
                        best_acc = res["acc"]
                        # DONE: Write best acc to TF variable
                        sess.run(
                            self.acc_assign_op,
                            feed_dict={
                                self.best_va_acc_in: best_acc
                            })
                        # Save the best model
                        self.saver_best.save(
                            sess, self.save_file_best,
                            write_meta_graph=False,
                        )

    def test(self, x_te, y_te):
        """Test routine"""

        with tf.Session() as sess:
            # Load the best model
            latest_checkpoint = tf.train.latest_checkpoint(
                self.config.save_dir)
            if tf.train.latest_checkpoint(self.config.save_dir) is not None:
                print("Restoring from {}...".format(
                    self.config.save_dir))
                self.saver_best.restore(
                    sess,
                    latest_checkpoint
                )

            # Test on the test data
            bs = config.batch_size
            num_test_b = len(x_te) // bs
            acc_list = []
            for idx_b in range(num_test_b):
                res = sess.run(
                    fetches={
                        "acc": self.acc,
                    },
                    feed_dict={
                        self.x_in: x_te[idx_b * bs: (idx_b + 1) * bs],
                        self.y_in: y_te[idx_b * bs: (idx_b + 1) * bs],
                    },
                )
                acc_list += [res["acc"]]
            res_acc = np.mean(acc_list)

            # Report (print) test result
            print("Test accuracy with the best model is {}".format(
                res_acc))
            # # Test on the test data
            # res = sess.run(
            #     fetches={
            #         "acc": self.acc,
            #     },
            #     feed_dict={
            #         self.x_in: x_te,
            #         self.y_in: y_te,
            #     },
            # )

            # # Report (print) test result
            # print("Test accuracy with the best model is {}".format(
            #     res["acc"]))





def main(config):
    """The main function."""

    # ----------------------------------------
    # Load cifar10 train data
    print("Reading training data...")
    x_trva, y_trva = TODO

    # ----------------------------------------
    # Load cifar10 test data
    print("Reading test data...")
    x_te, y_te = TODO

    # Randomly shuffle data and labels. IMPORANT: make sure the data and label
    # is shuffled with the same random indices so that they don't get mixed up!
    idx_shuffle = np.random.permutation(len(x_trva))
    x_trva = x_trva[idx_shuffle]
    y_trva = y_trva[idx_shuffle]

    # Change type to float32 and int64 since we are going to use that for
    # TensorFlow.
    x_trva = x_trva.astype("float32")
    y_trva = y_trva.astype("float32")

    # ----------------------------------------
    # Simply select the last 20% of the training data as validation dataset.
    num_tr = int(len(x_trva) * 0.8)

    x_tr = x_trva[:num_tr]
    x_va = x_trva[num_tr:]
    y_tr = y_trva[:num_tr]
    y_va = y_trva[num_tr:]

    # ----------------------------------------
    # Init network class
    mynet = MyNetwork(x_tr.shape, config)

    # ----------------------------------------
    # Train
    # Run training
    mynet.train(x_tr, y_tr, x_va, y_va)

    # ----------------------------------------
    # Test
    mynet.test(x_te, y_te)


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)

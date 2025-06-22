import os
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from DNNModel import D2NN as Forward
from DNNModel import D2NNOutput as Output
from MNISTPreprocess import dataset
# from FashionMNISTPreprocess import dataset
from Mask import create_mask
from ImshowConfusionMatirx import imshow_confusion_matrix
from ImshowEnergyDistribution import imshow_energy_distribution
import time


@tf.function
def func_one_hot(ms):
    return tf.vectorized_map(fn=one_hot, elems=ms, fallback_to_while_loop=True)


def one_hot(m):
    ans0 = tf.math.reduce_sum(tf.math.multiply(m, masks[0]))
    ans1 = tf.math.reduce_sum(tf.math.multiply(m, masks[1]))
    ans2 = tf.math.reduce_sum(tf.math.multiply(m, masks[2]))
    ans3 = tf.math.reduce_sum(tf.math.multiply(m, masks[3]))
    ans4 = tf.math.reduce_sum(tf.math.multiply(m, masks[4]))
    ans5 = tf.math.reduce_sum(tf.math.multiply(m, masks[5]))
    ans6 = tf.math.reduce_sum(tf.math.multiply(m, masks[6]))
    ans7 = tf.math.reduce_sum(tf.math.multiply(m, masks[7]))
    ans8 = tf.math.reduce_sum(tf.math.multiply(m, masks[8]))
    ans9 = tf.math.reduce_sum(tf.math.multiply(m, masks[9]))
    answers = tf.concat([[ans0], [ans1], [ans2], [ans3], [ans4],
                         [ans5], [ans6], [ans7], [ans8], [ans9]], axis=0)
    return answers


def MSE(y_pred, y_truth):
    return tf.math.reduce_mean(tf.keras.losses.MSE(y_true=y_truth, y_pred=y_pred))


def SCE(y_pred, y_truth):
    return tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_truth))


def energy_distribution(x):
    return x / np.sum(x, -1).reshape((-1, 1))


def imshow_trains(model_out, xs, ys, phase_input):
    fig = plt.figure(figsize=(12, 8))
    for pic in range(1):
        out = model_out(input_image=xs[pic], phase_input=phase_input)
        ans = tf.math.argmax(one_hot(out), axis=-1).numpy()
        ax = fig.add_subplot(1, 2, pic + 1)
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        fig15 = ax.get_figure()
        fig15.add_axes(ax_cb)
        ax.set_title("Truth: %d  Predicted: %d" % (tf.math.argmax(ys[pic]).numpy(), ans))
        if mask_border == 0:
            im = ax.imshow(out.numpy(), cmap="gray")
        else:
            im = ax.imshow((out.numpy())[mask_border:-mask_border, mask_border:-mask_border], cmap="gray")
        ax.axis('off')
        plt.colorbar(im, cax=ax_cb)
        ax_cb.yaxis.tick_right()
        edge = rang_size * (dimension - mask_border * 2) // base
        edge_color = "red"
        for number in range(10):
            rect = Rectangle(rect_location[number], edge[number], edge[number], edgecolor=edge_color,
                             fill=False, linestyle='--', linewidth=1)
            ax.add_patch(rect)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Phase 0")
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig6 = ax.get_figure()
    fig6.add_axes(ax_cb)
    im = ax.imshow(tf.sigmoid(phase_input[0]).numpy(), vmin=0, vmax=1, cmap='twilight')
    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    ax.axis('off')
    plt.show()


def train(layers, dim, phase_dim, ds, amp_or_phase):
    model = Forward(layers=layers, dim_in=dim, dim_out=dim, phase_dim=phase_dim, amp_or_phase=amp_or_phase,
                    mask_border=mask_border, rang_size=rang_size, lmb=lmb, ds=ds, phase_init=phase_init,
                    mask_on=mask_on, normalize=normalize)
    model_out = Output(layers=layers, dim_in=dim, dim_out=dim, phase_dim=phase_dim, lmb=lmb, ds=ds,
                       amp_or_phase=amp_or_phase)
    for ep in range(epochs):
        print("Epoch: %d / %d" % (ep + 1, epochs))
        train_batch = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size=batch_size)
        lr = lr_base * lr_decay ** ep
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        for i, examples in tqdm(enumerate(train_batch.take(total_batch))):
            xs, ys = examples['image'], examples['label']
            if is_view and i % view_every == 0:
                imshow_trains(model_out=model_out, xs=xs, ys=ys, phase_input=model.phase)
            with tf.GradientTape() as tape:
                output = model(input_image=xs)
                output_one_hot = func_one_hot(output)
                lossMSE = MSE(output, masks[tf.math.argmax(ys, axis=-1).numpy()])
                lossSCE = SCE(output_one_hot, ys)
                loss = tf.math.add(MSE_part * lossMSE, SCE_part * lossSCE)
            grads = tape.gradient(loss, [model.phase])
            optimizer.apply_gradients(grads_and_vars=zip(grads, [model.phase]))
        # truths, preds, _, _ = test(model)
        # acc = np.count_nonzero(truths == preds) / truths.shape[0] * 100
        # print("Accuracy: %.2f" % acc)
    return model


def test(model):
    test_batchs = test_dataset.batch(batch_size=batch_size)
    ground_truth_argmax = np.zeros((10000,), dtype=np.uint8)
    predicted_argmax = np.zeros((10000,), dtype=np.uint8)
    energy_sum = np.zeros((10000,), dtype=np.float32)
    predicted_energy_distribution = np.zeros((10000, 10), dtype=np.float32)
    for i, test_batch in tqdm(enumerate(test_batchs.take(total_batch))):
        xs, ys = test_batch['image'], test_batch['label']
        output = model(input_image=xs)
        energy_sum[i*batch_size:i*batch_size+batch_size] = tf.math.reduce_sum(output, axis=(-1, -2))
        batch_predicted_one_hot = func_one_hot(output)
        predicted_argmax[i*batch_size:i*batch_size+batch_size] = tf.math.argmax(batch_predicted_one_hot, axis=-1).numpy()
        ground_truth_argmax[i*batch_size:i*batch_size+batch_size] = np.argmax(ys, axis=-1)
        predicted_energy_distribution[i*batch_size:i*batch_size+batch_size] = batch_predicted_one_hot.numpy()
    return ground_truth_argmax, predicted_argmax, predicted_energy_distribution, energy_sum


if __name__ == "__main__":
    import datetime
    import pandas as pd

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    dimension = 200  # Number of pixels on each side of input plane
    phase_dimension = 1000  # Number of pixels on each side of phase modulation layer
    lmb = 532e-9  # Wavelength (um)
    MSE_part = 0.8  # loss = MSE loss * MSE_part + SCE loss * (1 - MSE_part)
    SCE_part = 0.2
    stddev = 0.  # phase errors
    phase_init = 0.

    mask_shape = '343'  # Mask shape (343, 3223, circle)
    lr_base = 0.1
    lr_decay = 0.99
    epochs = 10
    batch_size = 128
    buffer_size = 1000
    total_batch = 60000 // batch_size

    is_view = True  # If output intensity is needed
    view_every = 1000
    is_test = True
    is_save = True
    border = 0  # 0s around the input digit
    mask_border = 50  # 0s around the output plane
    mask_on = True  # If the optical field is covered with the mask
    normalize = True
    rang_size_i = 30
    rang_size = np.ones((10,)) * rang_size_i
    base = 512

    # Dataset
    (train_data, train_labels), (test_data, test_labels) = dataset(dimension=dimension, border=border, is_clip=False)
    # (train_data, train_labels), (test_data, test_labels) = dataset_8bit(dimension=dimension, border=border)
    # train_dataset = tf.data.Dataset.from_tensor_slices({"image": train_data[np.newaxis, 0], "label": train_labels[np.newaxis, 0]})
    train_dataset = tf.data.Dataset.from_tensor_slices({"image": train_data, "label": train_labels})
    test_dataset = tf.data.Dataset.from_tensor_slices({"image": test_data, "label": test_labels})

    # Mask
    masks, mask_location, rect_location = create_mask(shape=mask_shape, dim=dimension,
                                                      border=mask_border, rang_size=rang_size)
    # plt.imshow(np.sum(masks, axis=0))
    # plt.show()

    layers = 1  # Number of phase modulation layers
    # Diffraction distance
    d1 = 218e-3  # mm
    # d2 = 0e3  # mm
    d2 = 177e-3
    # d3 = 0.05
    ds = [d1, d2]
    amp_or_phase = ['phase']
    # Train
    DNN_model = train(layers=layers, dim=dimension, phase_dim=phase_dimension, ds=ds, amp_or_phase=amp_or_phase)

    date = datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')
    data = {'Parameter': ['Dimension', 'Phase dimension', 'Border', 'Mask border', 'Normalization',
                          'Mask', 'Rang size', 'Learning rate', 'Learning rate decay', 'Batch size',
                          'Epochs', 'MSE', 'SCE', 'Stddev', 'Phase initialization'],
            'Value': ['%d' % dimension, '%d' % phase_dimension, '%d' % border, '%d' % mask_border, str(normalize),
                      str(mask_on), '%d' % rang_size_i, '%.3f' % lr_base, '%.3f' % lr_decay, '%d' % batch_size,
                      '%d' % epochs, '%.3f' % MSE_part, '%.3f' % SCE_part, '%.3f' % stddev, '%.3f' % phase_init]}
    df = pd.DataFrame(data)

    # Test
    truths, preds, preds_energy_distribution, energy_all = test(DNN_model)
    acc = np.count_nonzero(truths == preds) / truths.shape[0] * 100
    power_efficiency = np.sum(preds_energy_distribution) / np.sum(energy_all)

    save_dir = "D:\YuHaonan\DNN/" + date + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_csv(save_dir + "parameter.txt", sep=' ', index=0, header=0)

    # Show Confusion Matrix and Energy Distribution
    imshow_confusion_matrix(truths=truths, preds=preds,
                            is_save=is_save, save_name=save_dir + "acc=%.2f" % acc)
    imshow_energy_distribution(truths=truths, preds=preds_energy_distribution,
                               is_save=is_save,
                               save_name=save_dir)  # + 'power efficiency=%.3f' % (1000 * power_efficiency))

    # Save the phase layer
    if is_save:
        np.save(save_dir + "phase.npy", tf.sigmoid(DNN_model.phase).numpy())
        # np.save(save_dir + "phase2.npy", tf.sigmoid(DNN_model.phase2).numpy())



import numpy as np

from sunyata.dataset.imdb import load_imdb
from sunyata.metric import *  # noqa
from sunyata.model import Model
from sunyata.node import *  # noqa
from sunyata.optim import *  # noqa


def make_conv(seq_len, in_dtype, vocab_size):
    def layer(channels):
        return SequenceSpec([
            ConvSpec(channels),
            GlobalBatchNormSpec(),
            ReLUSpec(),
            MaxPoolSpec(),
        ])

    seq = [
        InputSpec((seq_len,), in_dtype),
        EmbedSpec(vocab_size, 8),
    ]

    for i in range(8):
        seq.append(layer(8))

    seq += [
        FlattenSpec(),
        DenseSpec(1),
        SigmoidSpec(),
    ]

    return Model(SequenceSpec(seq))


num_epochs = 50
batch_size = 64

data, tf = load_imdb()
print(data[0][0].shape)
print(data[0][1].shape)

x_train, y_train = data[0]
seq_len = x_train.shape[1]
vocab_size = int(np.max(x_train)) + 1

model = make_conv(seq_len, x_train.dtype, vocab_size)

opt = NAG()

losses = [BinaryCrossEntropy()]
aux_metrics = [[BinaryAccuracy()]]

model.fit(data, opt, losses, aux_metrics, num_epochs, batch_size)

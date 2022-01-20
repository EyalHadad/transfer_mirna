from abc import ABC

from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, Embedding, MaxPooling1D, Flatten, LSTM, Input, \
    concatenate
from keras.models import Sequential, Model
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.regularizers import l1
from keras import layers, Model


def base_learning(my_shape):
    print("----- 4 Layers network------")
    model = Sequential()
    model.add(Dense(my_shape, input_dim=my_shape, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(my_shape / 2, activation='relu', kernel_constraint=maxnorm(3), activity_regularizer=l1(0.001),
                    kernel_regularizer=l1(0.001)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(my_shape / 4, activation='relu', kernel_constraint=maxnorm(3), activity_regularizer=l1(0.001),
                    kernel_regularizer=l1(0.001)))
    model.add(Dense(my_shape / 8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['acc'])
    return model


def cnn_learning2():
    # define model
    model = Sequential()
    model.add(Embedding(5, 100, input_length=130))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    print(model.summary())

    return model


def rnn_learning():
    # define model
    model = Sequential()
    model.add(Embedding(5, 100, input_length=130))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    # model.add(LSTM(100))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    print(model.summary())

    return model


def build_cnn_branch():
    inputs = Input(shape=(130,))
    y = Embedding(5, 100)(inputs)
    y = Conv1D(filters=32, kernel_size=8, activation='relu')(y)
    y = MaxPooling1D(pool_size=2)(y)
    y = Flatten()(y)
    y = Dense(10, activation='relu')(y)
    y = Dense(1, activation='sigmoid')(y)
    y = Model(inputs, y)
    return y


def miTrans_base_model(my_shape):
    print("----- 4 Layers network------")

    ann_branch = build_ann_branch(my_shape)
    cnn_branch = build_cnn_branch()
    combined = concatenate([ann_branch.output, cnn_branch.output])
    z = Dense(2, activation="relu")(combined)
    z = Dense(1, activation='sigmoid')(z)
    model = Model(inputs=[ann_branch.input, cnn_branch.input], outputs=z)
    # opt = Adam(lr=1e-3, decay=1e-3 / 200)
    # model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['acc'])
    print(model.summary())
    return model


class miTransfer(Model):
    def __init__(self, input_shape, vocab_size=5, maxlen=130, embed_dim=100):
        super(miTransfer, self).__init__()
        # CNN configuration
        self.cnn_input = Input(shape=(maxlen,))
        self.cnn_embeddings = Embedding(vocab_size, embed_dim)
        self.cnn_conv1 = Conv1D(filters=32, kernel_size=8, activation='relu')
        self.cnn_maxpool = MaxPooling1D(pool_size=2)
        self.cnn_flatten = Flatten()
        self.cnn_dense = Dense(10, activation='relu')
        self.cnn_output = Dense(1, activation='sigmoid')
        # ANN configuration
        self.ann_input = Input(shape=(input_shape,))
        self.ann_dropout = Dropout(rate=0.5)
        self.ann_dense1 = Dense(input_shape, activation='relu')
        self.ann_dense2 = Dense(input_shape / 2, activation='relu', kernel_constraint=maxnorm(3),
                                activity_regularizer=l1(0.001),kernel_regularizer=l1(0.001))
        self.ann_dense3 = Dense(input_shape / 4, activation='relu', kernel_constraint=maxnorm(3),
                                activity_regularizer=l1(0.001), kernel_regularizer=l1(0.001))
        self.ann_dense4 = Dense(input_shape / 8, activation='relu', kernel_constraint=maxnorm(3),
                                activity_regularizer=l1(0.001), kernel_regularizer=l1(0.001))
        self.ann_output = Dense(1, activation='sigmoid')

        # Merge configuration
        self.combined_layer = Dense(2, activation="relu")
        self.final_output = Dense(1, activation='sigmoid')


    def call(self, inputs, training=False):

        y= self.ann_input(inputs[0])
        y= self.ann_dense1(y)
        y= self.ann_dense2(y)
        y= self.ann_dropout(y)
        y= self.ann_dense3(y)
        y= self.ann_dropout(y)
        y= self.ann_dense4(y)
        y= self.ann_output(y)

        x = self.cnn_input(inputs[1])
        x = self.cnn_embeddings(x)
        x = self.cnn_conv1(x)
        x = self.cnn_maxpool(x)
        x = self.cnn_flatten(x)
        x = self.cnn_dense(x)
        x = self.cnn_output(x)

        z = self.combined_layer([x,y])
        z = self.final_output(z)

        return z


    def build_ann_branch(my_shape):
        inputs = Input(shape=(my_shape,))
        x = Dense(my_shape, activation='relu')(inputs)
        x = Dense(my_shape / 2, activation='relu', kernel_constraint=maxnorm(3), activity_regularizer=l1(0.001),
                  kernel_regularizer=l1(0.001))(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(my_shape / 4, activation='relu', kernel_constraint=maxnorm(3), activity_regularizer=l1(0.001),
                  kernel_regularizer=l1(0.001))(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(my_shape / 8, activation='relu', kernel_constraint=maxnorm(3), activity_regularizer=l1(0.001),
                  kernel_regularizer=l1(0.001))(x)
        x = Dense(1, activation='sigmoid')(x)
        x = Model(inputs, x)
        return x

    def build_cnn_branch():
        inputs = Input(shape=(130,))
        y = Embedding(5, 100)(inputs)
        y = Conv1D(filters=32, kernel_size=8, activation='relu')(y)
        y = MaxPooling1D(pool_size=2)(y)
        y = Flatten()(y)
        y = Dense(10, activation='relu')(y)
        y = Dense(1, activation='sigmoid')(y)
        y = Model(inputs, y)
        return y


def build_ann_branch(my_shape):
    inputs = Input(shape=(my_shape,))
    x = Dense(my_shape, activation='relu')(inputs)
    x = Dense(my_shape / 2, activation='relu', kernel_constraint=maxnorm(3), activity_regularizer=l1(0.001),
              kernel_regularizer=l1(0.001))(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(my_shape / 4, activation='relu', kernel_constraint=maxnorm(3), activity_regularizer=l1(0.001),
              kernel_regularizer=l1(0.001))(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(my_shape / 8, activation='relu', kernel_constraint=maxnorm(3), activity_regularizer=l1(0.001),
              kernel_regularizer=l1(0.001))(x)
    x = Dense(1, activation='sigmoid')(x)
    x = Model(inputs, x)
    return x

class miSmallTransfer(Model):
    def __init__(self, input_shape, vocab_size=5, maxlen=130, embed_dim=100):
        super(miSmallTransfer, self).__init__()
        # CNN configuration
        self.cnn_input = Input(shape=(maxlen,))
        self.cnn_embeddings = Embedding(vocab_size, embed_dim)
        self.cnn_conv1 = Conv1D(filters=32, kernel_size=8, activation='relu')
        self.cnn_maxpool = MaxPooling1D(pool_size=2)
        self.cnn_flatten = Flatten()
        self.cnn_dense = Dense(10, activation='relu')
        self.cnn_output = Dense(1, activation='sigmoid')
        # ANN configuration
        self.ann_input = Input(shape=(input_shape,))
        self.ann_dense1 = Dense(input_shape, activation='relu')
        self.ann_output = Dense(1, activation='sigmoid')

        # Merge configuration
        self.concatenate_layer = Dense(2, activation="relu")
        self.combined_layer = Dense(2, activation="relu")
        self.final_output = Dense(1, activation='sigmoid')


    def call(self, inputs, training=False):

        # y= self.ann_input(inputs[0])
        y= self.ann_dense1(inputs[0])
        y= self.ann_output(y)

        # x = self.cnn_input(inputs[1])
        x = self.cnn_embeddings(inputs[1])
        x = self.cnn_dense(x)
        x = self.cnn_output(x)
        x = layers.concatenate([x,y])
        z = self.combined_layer(x)
        z = self.final_output(z)

        return z

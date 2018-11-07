import keras.backend as K
import keras


class CustomConnections(keras.layers.Dense):
    def __init__(self, units, masks, step=0, **kwargs):
        super().__init__(units, **kwargs)
        self.masks = masks
        self.step = step

    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.masks[self.step])  # this is the only thing changed
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


class MaskAdvancer(keras.callbacks.Callback):
    def __init__(self, every: int):
        super().__init__()
        self.every = every

    def on_epoch_begin(self, epoch, logs=None):
        for layer in self.model.layers:
            if isinstance(layer, CustomConnections):
                layer.step = min(epoch // self.every, len(layer.masks) - 1)


from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf

class RoiPoolingConv(Layer):
    '''
    ROI pooling layer for 2D inputs.
    # Arguments
        pool_size: int
            Size of pooling region to use, e.g., pool_size = 7 will result in a 7x7 region.
        num_rois: int
            Number of regions of interest to be used.
        rois_mat: 2D Tensor
            Tensor containing RoIs, with shape `(num_rois, 4)` and format (x, y, w, h).
    # Input shape
        4D tensor with shape:
        `(batch_size, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        5D tensor with shape:
        `(batch_size, num_rois, pool_size, pool_size, channels)` if dim_ordering='tf'.
    '''

    def __init__(self, pool_size, num_rois, rois_mat, **kwargs):
        super(RoiPoolingConv, self).__init__(**kwargs)

        self.pool_size = pool_size
        self.num_rois = num_rois
        self.rois = rois_mat
        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {'channels_last', 'channels_first'}, "dim_ordering must be in {'channels_last', 'channels_first'}"

    def build(self, input_shape):
        self.nb_channels = input_shape[-1]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, img, mask=None):
        outputs = []

        for roi_idx in range(self.num_rois):
            x1 = self.rois[roi_idx, 0]
            y1 = self.rois[roi_idx, 1]
            w = self.rois[roi_idx, 2]
            h = self.rois[roi_idx, 3]

            x1 = K.cast(x1, 'int32')
            y1 = K.cast(y1, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            x2 = x1 + w
            y2 = y1 + h

            # Ensure the slicing does not go out of bounds
            x2 = tf.minimum(x2, K.shape(img)[2])
            y2 = tf.minimum(y2, K.shape(img)[1])

            roi = img[:, y1:y2, x1:x2, :]
            rs = tf.image.resize(roi, (self.pool_size, self.pool_size))
            outputs.append(rs)

        if not outputs:
            raise ValueError("No outputs generated from RoI pooling. Check RoI data and loop conditions.")
        else:
            final_output = tf.concat(outputs, axis=0)

        final_output = K.reshape(final_output, (-1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        return final_output


    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois,
                  'rois_mat': self.rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

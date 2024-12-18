###########################
#
# translation of pytorch version from shvit.py to tensorflow/keras
# (with help of keras- and pytorch docu, chatgpt, and try&error ...)
# 
###########################

import tensorflow as tf
import tensorflow.keras as keras

import keras.layers as KL

    

class GroupNorm(KL.Layer):
    """
    This implementation assumes the input tensor shape is [B, H, W, C], which is typical in TensorFlow/Keras, as opposed to [B, C, H, W] in PyTorch.
    The GroupNorm here normalizes over spatial dimensions (height and width) while keeping the channel dimension intact.
    mean and variance are computed across the spatial dimensions.
    """
    def __init__(self, num_channels, num_groups=1, **kwargs):
        super(GroupNorm, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.epsilon = 1e-5

    def call(self, inputs):
        # Reshape input to (B, H, W, C)
        inputs = tf.convert_to_tensor(inputs)
        B, H, W, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], self.num_channels
        
        # Reshape for group normalization
        inputs = tf.reshape(inputs, (B, H, W, self.num_groups, C // self.num_groups))
        
        # Calculate mean and variance for each group
        mean, variance = tf.nn.moments(inputs, axes=[1, 2, 4], keepdims=True)

        # Normalize
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)

        # Reshape back to original dimensions
        normalized = tf.reshape(normalized, (B, H, W, C))

        return normalized


 
class Conv2d_BN(KL.Layer):
    def __init__(self, a=0, filters=16, kernel_size=1, strides=1, padding='same', dilation_rate=1, groups=1, use_bn=True, activation="relu"):
        super(Conv2d_BN, self).__init__()
        self.conv = KL.Conv2D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=1, groups=1, use_bias=not use_bn)
        self.bn = KL.BatchNormalization() if use_bn else None
        self.activation = KL.Activation(activation) if activation else None

    def call(self, x, training=False):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x
    

# copy from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/helpers.py
def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

# translated with ChatGPT (original from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/squeeze_excite.py)
class SqueezeExcite(KL.Layer):
    def __init__(self, channels, rd_ratio=1/16, rd_channels=None, rd_divisor=8, add_maxpool=False,
                 bias=True, act_layer="relu", norm_layer=None, gate_layer="sigmoid"):
        super(SqueezeExcite, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(int(channels * rd_ratio), rd_divisor)
        
        self.fc1 = KL.Conv2D(filters=rd_channels, kernel_size=1, use_bias=bias)
        self.bn = norm_layer() if norm_layer else KL.Lambda(lambda x: x)
        self.act = KL.Activation(act_layer)
        self.fc2 = KL.Conv2D(filters=channels, kernel_size=1, use_bias=bias)
        self.gate = KL.Activation(gate_layer)

    def call(self, x):
        x_se = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * tf.reduce_max(x, axis=[1, 2], keepdims=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)
    

class PatchMerging(KL.Layer):
    """
    Initializes three Conv2d_BN layers and an activation function.
    The call method processes the input through these layers sequentially, applying ReLU activations and squeeze-and-excitation before the final convolution.
    """
    def __init__(self, dim, out_dim, train_bn=True, **kwargs):
        super(PatchMerging, self).__init__(**kwargs)
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(a=dim, filters=hid_dim, kernel_size=1, use_bn=train_bn)
        self.act = keras.activations.relu
        self.conv2 = Conv2d_BN(a=hid_dim, filters=hid_dim, kernel_size=3, strides=2, padding="same", groups=hid_dim, use_bn=train_bn)
        self.se = SqueezeExcite(channels=hid_dim, rd_ratio=0.25)
        self.conv3 = Conv2d_BN(a=hid_dim, filters=out_dim, kernel_size=1, use_bn=train_bn)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.se(x)
        x = self.conv3(x)
        return x


class Residual(KL.Layer):
    """
    Initialization: Takes a layer m and a dropout probability drop.

    Forward Pass:
        During training, if dropout is active, a random mask is applied to the output of m.
        If not in training mode, it simply adds the output of m to the input.

    Fusion:
        If m is a Conv2d_BN instance, it fuses the convolution and batch normalization layers.
        An identity tensor is created, padded, and added to the convolution weights.
    """
    def __init__(self, m, drop=0.0, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.m = m
        self.drop = drop

    def call(self, inputs, use_bn=None):
        if use_bn and self.drop > 0:
            # Generate a random mask for dropout
            rand_tensor = tf.random.uniform((tf.shape(inputs)[0], 1, 1, 1), 0, 1)
            mask = tf.cast(rand_tensor >= self.drop, tf.float32) / (1 - self.drop)
            return inputs + self.m(inputs) * mask
        else:
            return inputs + self.m(inputs)
        


class FFN(tf.keras.layers.Layer):
    """
    Initialization: Initializes two Conv2d_BN layers (pointwise convolutions) and a ReLU activation.
    Forward Pass: Applies the first convolution, then the ReLU activation, and finally the second convolution, returning the output.
    """
    def __init__(self, ed, h, train_bn=True, **kwargs):
        super(FFN, self).__init__(**kwargs)
        self.pw1 = Conv2d_BN(a=ed, filters=h, use_bn=train_bn)  # First pointwise convolution with BN
        self.act = keras.activations.relu  # ReLU activation
        self.pw2 = Conv2d_BN(a=h, filters=ed, use_bn=train_bn)  # Second pointwise convolution with BN

    def call(self, inputs):
        x = self.pw1(inputs)
        x = self.act(x)
        x = self.pw2(x)
        return x
    


class SHSA(KL.Layer):
    """Single-Head Self-Attention"""
    """
    Initialization: 
        Iitializes scaling factor, dimensions, normalization layer, and the query-key-value (QKV) convolutional layer, along with a projection layer.

    Forward Pass:
        Splits the input into two parts, applies normalization, and computes QKV.
        Flattens Q, K, and V, calculates the attention scores, applies softmax, and reshapes the result.
        Concatenates the processed part with the other part and passes it through the projection layer.
    """
    def __init__(self, dim, qk_dim, pdim, train_bn=True, **kwargs):
        super(SHSA, self).__init__(**kwargs)

        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = GroupNorm(num_channels=pdim)  # Assuming GroupNorm is defined
        self.qkv = Conv2d_BN(a=pdim, filters=qk_dim * 2 + pdim, use_bn=train_bn)  # Conv2d_BN layer
        self.proj = tf.keras.Sequential([
            keras.layers.ReLU(),
            Conv2d_BN(a=dim, filters=dim, use_bn=train_bn)  # Another Conv2d_BN layer
        ])

    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x1, x2 = tf.split(x, [self.pdim, self.dim - self.pdim], axis=-1)
        x1 = self.pre_norm(x1)
        
        qkv = self.qkv(x1)
        q, k, v = tf.split(qkv, [self.qk_dim, self.qk_dim, self.pdim], axis=-1)
        
        q = tf.reshape(q, (B, -1, self.qk_dim))
        k = tf.reshape(k, (B, -1, self.qk_dim))
        v = tf.reshape(v, (B, -1, self.pdim))

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)

        x1 = tf.reshape(tf.matmul(attn, v), (B, H, W, self.pdim))
        x = self.proj(tf.concat([x1, x2], axis=-1))

        return x



class BasicBlock(KL.Layer):
    """
    Initialization:
        For "s" (later stages): Initializes convolution, self-attention mixer, and feed-forward network (FFN) wrapped in residuals.
        For "i" (early stages): Initializes convolution and FFN as before but uses an identity layer for the mixer.

    Forward Pass:
        Calls the convolution layer, the mixer, and the feed-forward network sequentially, returning the output.
    """
    def __init__(self, dim, qk_dim, pdim, block_type, train_bn=True, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        if block_type == "s":  # for later stages
            self.conv = Residual(Conv2d_BN(a=dim, filters=dim, kernel_size=3, strides=1, padding="same", groups=dim, use_bn=train_bn))
            self.mixer = Residual(SHSA(dim=dim, qk_dim=qk_dim, pdim=pdim, train_bn=train_bn))
            self.ffn = Residual(FFN(ed=dim, h=int(dim * 2), train_bn=train_bn))
        elif block_type == "i":  # for early stages
            self.conv = Residual(Conv2d_BN(a=dim, filters=dim, kernel_size=3, strides=1, padding="same", groups=dim, use_bn=train_bn))
            self.mixer = KL.Layer()  # Identity layer
            self.ffn = Residual(FFN(ed=dim, h=int(dim * 2), train_bn=train_bn))

    def call(self, x):
        return self.ffn(self.mixer(self.conv(x)))
    



#################################################################
def shvit_s4(input_image, architecture, stage5=False, train_bn=True):
    """
    Patch Embedding: The patch_embed is defined as a Keras Sequential model containing convolutional layers.
    SHViT Blocks: Blocks are created similarly, using the BasicBlock class.
    Freezing Layers: The _freeze_stages method disables training for certain layers.
    Weight Initialization: Placeholder for logic to initialize weights from pretrained models.
    Forward Pass: The call method processes the input through the patch embedding and blocks, collecting outputs at each stage.
    """
    embed_dim = [224, 336, 448]
    partial_dim = [48, 72, 96]    # partial_dim = r*embed_dim with r=1/4.67
    depth = [4, 7, 6]
    types = ["i", "s", "s"]

    qk_dim = [16, 16, 16]
    down_ops = [['subsample', 2], ['subsample', 2], ['']]
    #pretrained = None
    #distillation = False

    blocks1 = []
    blocks2 = []
    blocks3 = []
    outs = []
    outs.append(None) # C1 (dummy, not used in MaskRCNN)

    x = input_image
    # 16x16 Overlap PatchEmbed (Fig 2 and Fig 5), in SHViT-pytorch version of the code, each Conv2D is followed with BatchNormalization 
    x = Conv2d_BN(filters=embed_dim[0]//8, kernel_size=3, strides=2, padding="same", use_bn=train_bn, activation="ReLU")(x)
    x = Conv2d_BN(filters=embed_dim[0]//4, kernel_size=3, strides=2, padding="same", use_bn=train_bn, activation="ReLU")(x)
    x = Conv2d_BN(filters=embed_dim[0]//2, kernel_size=3, strides=2, padding="same", use_bn=train_bn, activation="ReLU")(x)
    outs.append(x) # C2
    x = Conv2d_BN(filters=embed_dim[0]//1, kernel_size=3, strides=2, padding="same", use_bn=train_bn, activation="ReLU")(x)
      

    for i, (ed, kd, pd, dpth, do, t) in enumerate(zip(embed_dim, qk_dim, partial_dim, depth, down_ops, types)):
        for d in range(dpth):
            eval("blocks" + str(i+1)).append(BasicBlock(ed, kd, pd, t))
        if do[0] == "subsample":
            # Build SHViT downsample block  
            blk = eval("blocks" + str(i+2))
            blk.append(keras.Sequential([
                        Residual(Conv2d_BN(a=embed_dim[i], filters=embed_dim[i], kernel_size=3, strides=1, padding="same", groups=embed_dim[i], use_bn=train_bn)),
                        Residual(FFN(ed=embed_dim[i], h=int(embed_dim[i] * 2), train_bn=train_bn)),
                    ]))
            blk.append(PatchMerging(dim=embed_dim[i], out_dim=embed_dim[i + 1], train_bn=train_bn))
            blk.append(keras.Sequential([
                        Residual(Conv2d_BN(a=embed_dim[i + 1], filters=embed_dim[i + 1], kernel_size=3, strides=1, padding="same", groups=embed_dim[i + 1], use_bn=train_bn)),
                        Residual(FFN(ed=embed_dim[i + 1], h=int(embed_dim[i + 1] * 2), train_bn=train_bn)),
                    ]))

    blocks1 = tf.keras.Sequential(blocks1, name="blocks1_i")
    blocks2 = tf.keras.Sequential(blocks2, name="blocks2_s")
    blocks3 = tf.keras.Sequential(blocks3, name="blocks3_s")

    x = blocks1(x)
    outs.append(x) # C3
    x = blocks2(x)
    outs.append(x) # C4
    x = blocks3(x)
    outs.append(x) # C5

    return outs

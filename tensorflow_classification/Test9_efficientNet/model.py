import math
from typing import Union

from tensorflow.keras import layers, Model


# Inicializador de kernel para camadas Conv2D
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',  # Utiliza a estratégia VarianceScaling para inicialização
    'config': {
        'scale': 2.0,                 # Fator de escala para a variância da distribuição
        'mode': 'fan_out',            # Modo de cálculo de fan-out (número de unidades de saída)
        'distribution': 'truncated_normal'  # Distribuição utilizada para inicialização
    }
}

# Inicializador de kernel para camadas Dense
DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',  # Utiliza a estratégia VarianceScaling para inicialização
    'config': {
        'scale': 1. / 3.,              # Fator de escala para a variância da distribuição
        'mode': 'fan_out',            # Modo de cálculo de fan-out (número de unidades de saída)
        'distribution': 'uniform'      # Distribuição utilizada para inicialização
    }
}


# Uma função utilitária que calcula o zero-padding necessário para convolução 2D com downsampling.
def correct_pad(input_size: Union[int, tuple], kernel_size: int):
    """Retorna uma tupla para zero-padding para convolução 2D com downsampling.

    Argumentos:
      input_size: Tamanho do tensor de entrada.
      kernel_size: Um inteiro ou tupla/lista de 2 inteiros.

    Retorna:
      Uma tupla.
    """

    # Se o tamanho de entrada for um número inteiro, converte para uma tupla (height, width)
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    # Converte o tamanho do kernel para uma tupla (kernel_height, kernel_width)
    kernel_size = (kernel_size, kernel_size)

    # Calcula os ajustes necessários com base na paridade do tamanho de entrada e do kernel
    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    # Calcula o zero-padding correto para a altura e largura
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    # Retorna uma tupla contendo as informações de zero-padding para a convolução 2D
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


# Esta função implementa um bloco residual invertido, um componente fundamental da arquitetura EfficientNet.
def block(inputs,
          activation: str = "swish",
          drop_rate: float = 0.,
          name: str = "",
          input_channel: int = 32,
          output_channel: int = 16,
          kernel_size: int = 3,
          strides: int = 1,
          expand_ratio: int = 1,
          use_se: bool = True,
          se_ratio: float = 0.25):
    """An inverted residual block.

      Argumentos:
          inputs: tensor de entrada.
          activation: função de ativação.
          drop_rate: float entre 0 e 1, fração das unidades de entrada a serem descartadas.
          name: string, rótulo do bloco.
          input_channel: inteiro, número de filtros de entrada.
          output_channel: inteiro, número de filtros de saída.
          kernel_size: inteiro, a dimensão da janela de convolução.
          strides: inteiro, o passo da convolução.
          expand_ratio: inteiro, coeficiente de escala para os filtros de entrada.
          use_se: se deve usar se.
          se_ratio: float entre 0 e 1, fração para reduzir os filtros de entrada.

      Retorna:
          tensor de saída para o bloco.
    """
    # Fase de Expansão
    filters = input_channel * expand_ratio
    if expand_ratio != 1:
        # Aplica convolução 1x1 para expandir o número de canais (filtros)
        x = layers.Conv2D(filters=filters,
                          kernel_size=1,
                          padding="same",
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=name + "expand_conv")(inputs)
        x = layers.BatchNormalization(name=name + "expand_bn")(x)
        x = layers.Activation(activation, name=name + "expand_activation")(x)
    else:
        # Se não houver expansão, mantém a entrada inalterada
        x = inputs

    # Convolução Profunda
    if strides == 2:
        # Adiciona zero-padding para lidar com o downsampling
        x = layers.ZeroPadding2D(padding=correct_pad(filters, kernel_size),
                                 name=name + "dwconv_pad")(x)

    # Aplica convolução depthwise
    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=strides,
                               padding="same" if strides == 1 else "valid",
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + "dwconv")(x)
    x = layers.BatchNormalization(name=name + "bn")(x)
    x = layers.Activation(activation, name=name + "activation")(x)

    # Squeeze-and-Excitation (SE)
    if use_se:
        filters_se = int(input_channel * se_ratio)
        # Global Average Pooling para reduzir a dimensionalidade espacial
        se = layers.GlobalAveragePooling2D(name=name + "se_squeeze")(x)
        se = layers.Reshape((1, 1, filters), name=name + "se_reshape")(se)
        # Redução de dimensionalidade linear
        se = layers.Conv2D(filters=filters_se,
                           kernel_size=1,
                           padding="same",
                           activation=activation,
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + "se_reduce")(se)
        # Expansão de dimensionalidade linear
        se = layers.Conv2D(filters=filters,
                           kernel_size=1,
                           padding="same",
                           activation="sigmoid",
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + "se_expand")(se)
        # Aplica o bloco SE ao tensor de entrada
        x = layers.multiply([x, se], name=name + "se_excite")

    # Fase de Saída
    # Aplica convolução 1x1 para projetar de volta ao número desejado de canais
    x = layers.Conv2D(filters=output_channel,
                      kernel_size=1,
                      padding="same",
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=name + "project_conv")(x)
    x = layers.BatchNormalization(name=name + "project_bn")(x)

    # Adiciona a entrada original se o número de canais não mudou e o stride é 1
    if strides == 1 and input_channel == output_channel:
        if drop_rate > 0:
            # Aplica dropout se especificado
            x = layers.Dropout(rate=drop_rate,
                               noise_shape=(None, 1, 1, 1),  # máscara de dropout binário
                               name=name + "drop")(x)
        # Adiciona a entrada original para a saída
        x = layers.add([x, inputs], name=name + "add")

    return x


# Esta função é a principal responsável pela criação da arquitetura EfficientNet com base nos parâmetros fornecidos.
def efficient_net(width_coefficient,
                  depth_coefficient,
                  input_shape=(224, 224, 3),
                  dropout_rate=0.2,
                  drop_connect_rate=0.2,
                  activation="swish",
                  model_name="efficientnet",
                  include_top=True,
                  num_classes=1000):
    """Instantiates the EfficientNet architecture using given scaling coefficients.

      Reference:
      - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
          https://arxiv.org/abs/1905.11946) (ICML 2019)

      Optionally loads weights pre-trained on ImageNet.
      Note that the data format convention used by the model is
      the one specified in your Keras config at `~/.keras/keras.json`.

      Arguments:
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        input_shape: tuple, default input image shape(not including the batch size).
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        activation: activation function.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        num_classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

      Returns:
        A `keras.Model` instance.
    """

    # kernel_size, repeats, in_channel, out_channel, exp_ratio, strides, SE
    block_args = [[3, 1, 32, 16, 1, 1, True],
                  [3, 2, 16, 24, 6, 2, True],
                  [5, 2, 24, 40, 6, 2, True],
                  [3, 3, 40, 80, 6, 2, True],
                  [5, 3, 80, 112, 6, 1, True],
                  [5, 4, 112, 192, 6, 2, True],
                  [3, 1, 192, 320, 6, 1, True]]

    # Funções auxiliares para arredondar o número de filtros e repetições com base nos coeficientes de largura e profundidade.
    def round_filters(filters, divisor=8):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    img_input = layers.Input(shape=input_shape)

    # data preprocessing
    x = layers.experimental.preprocessing.Rescaling(1. / 255.)(img_input)
    x = layers.experimental.preprocessing.Normalization()(x)

    # first conv2d
    x = layers.ZeroPadding2D(padding=correct_pad(input_shape[:2], 3),
                             name="stem_conv_pad")(x)
    x = layers.Conv2D(filters=round_filters(32),
                      kernel_size=3,
                      strides=2,
                      padding="valid",
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation(activation, name="stem_activation")(x)

    # build blocks
    b = 0
    num_blocks = float(sum(round_repeats(i[1]) for i in block_args))
    for i, args in enumerate(block_args):
        assert args[1] > 0
        # Update block input and output filters based on depth multiplier.
        args[2] = round_filters(args[2])  # input_channel
        args[3] = round_filters(args[3])  # output_channel

        for j in range(round_repeats(args[1])):
            x = block(x,
                      activation=activation,
                      drop_rate=drop_connect_rate * b / num_blocks,
                      name="block{}{}_".format(i + 1, chr(j + 97)),
                      kernel_size=args[0],
                      input_channel=args[2] if j == 0 else args[3],
                      output_channel=args[3],
                      expand_ratio=args[4],
                      strides=args[5] if j == 0 else 1,
                      use_se=args[6])
            b += 1

    # build top
    x = layers.Conv2D(round_filters(1280),
                      kernel_size=1,
                      padding="same",
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name="top_conv")(x)
    x = layers.BatchNormalization(name="top_bn")(x)
    x = layers.Activation(activation, name="top_activation")(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="top_dropout")(x)
        x = layers.Dense(units=num_classes,
                         activation="softmax",
                         kernel_initializer=DENSE_KERNEL_INITIALIZER,
                         name="predictions")(x)

    model = Model(img_input, x, name=model_name)

    return model


def efficientnet_b0(num_classes=1000,
                    include_top=True,
                    input_shape=(224, 224, 3)):
    # https://storage.googleapis.com/keras-applications/efficientnetb0.h5
    return efficient_net(width_coefficient=1.0,
                         depth_coefficient=1.0,
                         input_shape=input_shape,
                         dropout_rate=0.2,
                         model_name="efficientnetb0",
                         include_top=include_top,
                         num_classes=num_classes)


def efficientnet_b1(num_classes=1000,
                    include_top=True,
                    input_shape=(240, 240, 3)):
    # https://storage.googleapis.com/keras-applications/efficientnetb1.h5
    return efficient_net(width_coefficient=1.0,
                         depth_coefficient=1.1,
                         input_shape=input_shape,
                         dropout_rate=0.2,
                         model_name="efficientnetb1",
                         include_top=include_top,
                         num_classes=num_classes)


def efficientnet_b2(num_classes=1000,
                    include_top=True,
                    input_shape=(260, 260, 3)):
    # https://storage.googleapis.com/keras-applications/efficientnetb2.h5
    return efficient_net(width_coefficient=1.1,
                         depth_coefficient=1.2,
                         input_shape=input_shape,
                         dropout_rate=0.3,
                         model_name="efficientnetb2",
                         include_top=include_top,
                         num_classes=num_classes)


def efficientnet_b3(num_classes=1000,
                    include_top=True,
                    input_shape=(300, 300, 3)):
    # https://storage.googleapis.com/keras-applications/efficientnetb3.h5
    return efficient_net(width_coefficient=1.2,
                         depth_coefficient=1.4,
                         input_shape=input_shape,
                         dropout_rate=0.3,
                         model_name="efficientnetb3",
                         include_top=include_top,
                         num_classes=num_classes)


def efficientnet_b4(num_classes=1000,
                    include_top=True,
                    input_shape=(380, 380, 3)):
    # https://storage.googleapis.com/keras-applications/efficientnetb4.h5
    return efficient_net(width_coefficient=1.4,
                         depth_coefficient=1.8,
                         input_shape=input_shape,
                         dropout_rate=0.4,
                         model_name="efficientnetb4",
                         include_top=include_top,
                         num_classes=num_classes)


def efficientnet_b5(num_classes=1000,
                    include_top=True,
                    input_shape=(456, 456, 3)):
    # https://storage.googleapis.com/keras-applications/efficientnetb5.h5
    return efficient_net(width_coefficient=1.6,
                         depth_coefficient=2.2,
                         input_shape=input_shape,
                         dropout_rate=0.4,
                         model_name="efficientnetb5",
                         include_top=include_top,
                         num_classes=num_classes)


def efficientnet_b6(num_classes=1000,
                    include_top=True,
                    input_shape=(528, 528, 3)):
    # https://storage.googleapis.com/keras-applications/efficientnetb6.h5
    return efficient_net(width_coefficient=1.8,
                         depth_coefficient=2.6,
                         input_shape=input_shape,
                         dropout_rate=0.5,
                         model_name="efficientnetb6",
                         include_top=include_top,
                         num_classes=num_classes)


def efficientnet_b7(num_classes=1000,
                    include_top=True,
                    input_shape=(600, 600, 3)):
    # https://storage.googleapis.com/keras-applications/efficientnetb7.h5
    return efficient_net(width_coefficient=2.0,
                         depth_coefficient=3.1,
                         input_shape=input_shape,
                         dropout_rate=0.5,
                         model_name="efficientnetb7",
                         include_top=include_top,
                         num_classes=num_classes)

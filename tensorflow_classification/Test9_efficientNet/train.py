import os
import sys
import math
import datetime

import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from model import efficientnet_b0 as create_model
from utils import generate_ds

assert tf.version.VERSION >= "2.15.0", "version of tf must greater/equal than 2.4.0"


def main():
    data_root = "./../../data_set/portuguese_food_data/portuguese_food_photos"  # get data root path
    
    # Cria o diretório para salvar os pesos do modelo, se não existir
    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights")

    # Tamanhos de imagem para diferentes modelos EfficientNet
    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}

    num_model = "B0"  # Escolhe o modelo EfficientNet B0
    im_height = im_width = img_size[num_model]  # Altura e largura da imagem
    batch_size = 16  # Tamanho do lote para treinamento
    epochs = 30  # Número de épocas de treinamento
    num_classes = 23  # Número de classes no conjunto de dados
    freeze_layers = True  # Se as camadas inferiores do modelo devem ser congeladas
    initial_lr = 0.01  # Taxa de aprendizado inicial

    # Diretório para os logs
    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    # Geração do conjunto de dados de treinamento e validação
    train_ds, val_ds = generate_ds(data_root, im_height, im_width, batch_size)

    # Criação do modelo EfficientNet
    model = create_model(num_classes=num_classes)

    # Carrega os pesos pré-treinados
    pre_weights_path = './efficientnetb0.h5'
    assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

    # Congela as camadas inferiores do modelo, se configurado
    if freeze_layers:
        unfreeze_layers = ["top_conv", "top_bn", "predictions"]
        for layer in model.layers:
            if layer.name not in unfreeze_layers:
                layer.trainable = False
            else:
                print("training {}".format(layer.name))

    model.summary() # Exibe um resumo do modelo

    # Função para ajustar a taxa de aprendizado durante o treinamento
    def scheduler(now_epoch):
        end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
        new_lr = rate * initial_lr

        # Grava a taxa de aprendizado no TensorBoard
        with train_writer.as_default():
            tf.summary.scalar('learning rate', data=new_lr, step=epoch)

        return new_lr

    # Configuração do otimizador e funções de perda e métricas
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    # Definição da etapa de treinamento usando tf.function para otimizar o desempenho
    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            output = model(train_images, training=True)
            loss = loss_object(train_labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(train_labels, output)

    # Definição da etapa de validação usando tf.function
    @tf.function
    def val_step(val_images, val_labels):
        output = model(val_images, training=False)
        loss = loss_object(val_labels, output)

        val_loss(loss)
        val_accuracy(val_labels, output)

    best_val_acc = 0.  # Inicializa a melhor precisão de validação
    for epoch in range(epochs):
        # Zera as métricas de treinamento e validação no início de cada época
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        train_bar = tqdm(train_ds, file=sys.stdout)
        for images, labels in train_bar:
            train_step(images, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

        # Atualiza a taxa de aprendizado
        optimizer.learning_rate = scheduler(epoch)

        # validate
        val_bar = tqdm(val_ds, file=sys.stdout)
        for images, labels in val_bar:
            val_step(images, labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())
        # Grava as métricas de treinamento no TensorBoard
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), epoch)

        # Grava as métricas de validação no TensorBoard
        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), epoch)
            tf.summary.scalar("accuracy", val_accuracy.result(), epoch)

        # Salva apenas os melhores pesos
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            save_name = "./save_weights/efficientnet.ckpt"
            model.save_weights(save_name, save_format="tf")
    
    # Avaliação mais detalhada após o treinamento
    try:
        evaluate_model(model, val_ds)
    except Exception as e:
        pass

def evaluate_model(model, val_ds):
    true_labels = []
    predicted_labels = []

    for images, labels in val_ds:
        predictions = model.predict(images)
        predicted_labels.extend(tf.argmax(predictions, axis=1))
        true_labels.extend(labels)

    # Calcula a matriz de confusão
    confusion = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(confusion)

    # Calcula precisão, recall, F1-score e acurácia
    report = classification_report(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    print("Classification Report:")
    print(report)
    
    print("Accuracy:", accuracy)

if __name__ == '__main__':
    main()

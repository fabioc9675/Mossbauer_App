import tensorflow as tf
import numpy as np

# El modelo espera una lista con 512 intensidades del espectro


def Mossbauer_ann(model, espectro_list):
    # Cargar el modelo

    espectro = [espectro_list]

    categ = ["Otro", "Hematita", "Magnetita"]
    y = np.argmax(model.predict(espectro), axis=-1)

    return categ[y[0]]


if __name__ == '__main__':

    trained_model = tf.keras.models.load_model(
        "Mossbauer_model_2.h5", compile=False)

    print("Inicio de prediccion")

    spec_hema = -np.loadtxt('Hematita 15.txt', dtype=float)[:, 1] / 10
    spec_hema = spec_hema.tolist()
    spec_magn = -np.loadtxt('Magnetita 8.txt', dtype=float)[:, 1] / 10
    spec_magn = spec_magn.tolist()
    spec_otro = -np.loadtxt('Epidote.txt', dtype=float)[:, 1] / 10
    spec_otro = spec_otro.tolist()

    print(Mossbauer_ann(trained_model, spec_hema))
    print(Mossbauer_ann(trained_model, spec_magn))
    print(Mossbauer_ann(trained_model, spec_otro))

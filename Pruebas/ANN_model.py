import tensorflow.keras as kr
import numpy as np


def Mossbauer_ann(txt_espectro):
    # Cargar el modelo
    trained_model = kr.models.load_model("Mossbauer_model")

    # Carga el espectro y añade un valor para completar los 512 valores
    espectro = np.loadtxt(txt_espectro)

    espectro = np.append(espectro, espectro[-1])
    espectro = [espectro.tolist()]

    print(espectro)
    categ = ["Otro", "Hematita", "Magnetita"]
    y = np.argmax(trained_model.predict(espectro), axis=-1)

    return categ[y[0]]


if __name__ == '__main__':

    print("Inicio de prediccion")

    print(Mossbauer_ann('Hematita 15.txt'))
    print(Mossbauer_ann('Magnetita 8.txt'))
    print(Mossbauer_ann('Epidote.txt'))

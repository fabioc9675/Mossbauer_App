from typing import Optional
from django.conf import settings
from django.template.response import TemplateResponse
from django.utils.datastructures import MultiValueDictKeyError

from django.core.files.storage import FileSystemStorage

import tensorflow as tf
import numpy as np
import os

from .utils import get_plot


class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name


def index(request):
    message = ""
    prediction = ""
    fss = CustomFileSystemStorage()
    try:

        # load model
        model = tf.keras.models.load_model(
            os.getcwd() + os.path.join(os.sep, "model", "Mossbauer_model.h5"))

        spectrum = request.FILES["file"]
        print("Name", spectrum.file)
        _spectrum = fss.save(spectrum.name, spectrum)
        path = str(settings.MEDIA_ROOT) + "/" + spectrum.name
        # read the spectrum
        spec_ = -np.loadtxt(path, dtype=float)[:, 1] / 10
        spec_graph = - spec_
        spec_ = spec_.tolist()



        spectrum_pred = [spec_]
        categ = ["Otro", "Hematita", "Magnetita"]
        result = np.argmax(model.predict(spectrum_pred), axis=-1)

        print("Prediction: " + str(np.argmax(result)))

        prediction = categ[result[0]]

        return TemplateResponse(
            request,
            "index.html",
            {
                "message": message,
                "spectrum": spectrum_pred,
                "prediction": prediction
            }
        )
    except MultiValueDictKeyError:
        return TemplateResponse(
            request,
            "index.html",
            {
                "message": "No File Selected"
            },
        )

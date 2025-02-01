from itertools import product
import logging
import os
import shutil

from interface.cnn import CNN, define_transforms, read_images


def clean_models_directory(directory_path):
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        logging.info(f"Conteúdo da pasta {directory_path} foi apagado com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao tentar limpar a pasta {directory_path}: {e}")




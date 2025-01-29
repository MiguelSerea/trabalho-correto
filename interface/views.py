import os
import json
import logging
from itertools import product
import threading
import time
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from threading import Lock
from multiprocessing import Queue
from joblib import Parallel, delayed
import torch
from .cnn import CNN, define_transforms, read_images

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Instâncias de queue e lock para controle de concorrência nos processos
queue = Queue()
queue_lock = Lock()

def index_view(request):
    """Renderiza o template principal."""
    return render(request, 'index.html')

@csrf_exempt
def start_training(request):
    """Inicia o processo de treinamento quando o botão é clicado."""
    if request.method == 'POST':
        threading.Thread(target=run_training_process).start()
        return JsonResponse({"message": "Training started"}, status=200)
    return JsonResponse({"error": "Method not allowed"}, status=405)

def run_training_process():
    """Executa o processo de treinamento em uma thread separada."""
    try:
        # Configuração dos transformadores de dados para treinamento, validação e teste
        data_transforms = define_transforms(224, 224)
        train_data, validation_data, test_data = read_images(data_transforms)
        cnn_model = CNN(train_data, validation_data, test_data, batch_size=8)
        
        model_name = "resnet18"
        epochs = 20
        learning_rate = 0.0001
        weight_decay = 0.0001
        replications = 10

        # Diretório para salvar os modelos treinados
        save_directory = 'v1/modelos'
        os.makedirs(save_directory, exist_ok=True)

        # Treinamento do modelo
        average_accuracy, best_replication = cnn_model.create_and_train_cnn(
            model_name, epochs, learning_rate, weight_decay, replications, save_directory
        )
        logging.info(f"Training completed. Average accuracy: {average_accuracy}, Best replication: {best_replication}")

    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)

@csrf_exempt
def centralized_multiprocess_view(request):
    """Inicia treinamento multiprocessado centralizado (salva em v2/modelos)."""
    if request.method == 'POST':
        threading.Thread(target=run_centralized_multiprocess).start()
        return JsonResponse({"message": "Centralized multiprocess training started."}, status=200)
    return JsonResponse({"error": "Method not allowed"}, status=405)

def run_centralized_multiprocess():
    """Executa treinamento multiprocessado centralizado."""
    try:
        data_transforms = define_transforms(224, 224)
        train_data, validation_data, test_data = read_images(data_transforms)
        cnn = CNN(train_data, validation_data, test_data, 8)

        save_directory = "v2/modelos"
        os.makedirs(save_directory, exist_ok=True)

        replications = 2
        model_names = ['alexnet']
        epochs = [10]
        learning_rates = [0.001]
        weight_decays = [0, 0.0001]

        combinations = list(product(model_names, epochs, learning_rates, weight_decays))
        combinations_with_cnn = [
            (model_name, epoch, learning_rate, weight_decay, replications, cnn)
            for model_name, epoch, learning_rate, weight_decay in combinations
        ]

        results_file = os.path.join(save_directory, "results.txt")
        with open(results_file, "a") as file:
            results = Parallel(n_jobs=2, backend="multiprocessing")(
                delayed(process_combination)(args) for args in combinations_with_cnn
            )
            file.writelines(results)
    except Exception as e:
        logging.error(f"Error during centralized multiprocess training: {e}", exc_info=True)

@csrf_exempt
def centralized_uniprocess_view(request):
    """Inicia treinamento centralizado uniprocesso (salva em v1/modelos)."""
    if request.method == 'POST':
        threading.Thread(target=run_centralized_uniprocess).start()
        return JsonResponse({"message": "Centralized uniprocess training started."}, status=200)
    return JsonResponse({"error": "Method not allowed"}, status=405)

def run_centralized_uniprocess():
    """Executa treinamento centralizado uniprocesso."""
    try:
        data_transforms = define_transforms(224, 224)
        train_data, validation_data, test_data = read_images(data_transforms)
        cnn = CNN(train_data, validation_data, test_data, 8)

        save_directory = "v1/modelos"
        os.makedirs(save_directory, exist_ok=True)

        replications = 2
        model_names = ['alexnet']
        epochs = [10]
        learning_rates = [0.001]
        weight_decays = [0, 0.0001]

        combinations = list(product(model_names, epochs, learning_rates, weight_decays))
        combinations_with_cnn = [
            (model_name, epoch, learning_rate, weight_decay, replications, cnn)
            for model_name, epoch, learning_rate, weight_decay in combinations
        ]

        results_file = os.path.join(save_directory, "results.txt")
        with open(results_file, "a") as file:
            for args in combinations_with_cnn:
                result = process_combination(args)
                file.write(result)
    except Exception as e:
        logging.error(f"Error during centralized uniprocess training: {e}", exc_info=True)

def process_combination(args):
    """Processa uma combinação de parâmetros de treino."""
    model_name, epochs, learning_rate, weight_decay, replications, cnn = args

    model = cnn.create_model(model_name)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    try:
        start_time = time.time()
        for replication in range(replications):
            cnn.train_model(
                model,
                cnn.train_loader,
                optimizer,
                criterion,
                model_name,
                epochs,
                learning_rate,
                weight_decay,
                replication,
    
            )
        end_time = time.time()
        
        average_accuracy, better_replication = cnn.create_and_train_cnn(
            model_name, epochs, learning_rate, weight_decay, replications, save_directory=""
        )

        duration = end_time - start_time
        hours, rem = divmod(duration, 3600)
        minutes, seconds = divmod(rem, 60)
        milliseconds = int(duration * 1000 % 1000)

        result_text = (
            f"Combination:\n"
            f"\tModel: {model_name} - Epochs: {epochs} - LR: {learning_rate} - WD: {weight_decay}\n"
            f"\tAverage Accuracy: {average_accuracy}\n"
            f"\tBetter Replication: {better_replication}\n"
            f"\tExecution Time: {int(hours)}:{int(minutes)}:{int(seconds)}:{milliseconds:03}\n\n"
        )
        logging.info(result_text)
        return result_text
    except Exception as err:
        logging.error(f"Error in processing combination for model {model_name}: {err}", exc_info=True)
        return f"Error in processing combination for model {model_name}."

@csrf_exempt
def start_combinations(request):
    """Envia mensagem de status quando não há tarefas para o processamento descentralizado."""
    if request.method == 'POST':
        with queue_lock:
            if queue.empty():
                return JsonResponse({"message": "Não há tarefas na fila para processamento descentralizado."}, status=200)
            else:
                num_cores = request.POST.get('num_cores', 1)  # ou um argumento apropriado
                combinations = []
                
                for _ in range(int(num_cores)):
                    if queue.empty():
                        break
                    combinations.append(queue.get())
                
                return JsonResponse({"message": f"{len(combinations)} combinações enviadas para processamento."}, status=200)

    return JsonResponse({"error": "Método não permitido."}, status=405)

def combine():
    """Popula combinações na fila para processamento."""
    model_names = ['alexnet', 'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet101', 'vgg11', 'vgg19']
    epochs_list = [10, 20]
    learning_rates = [0.001, 0.0001, 0.00001]
    weight_decays = [0, 0.0001]

    combinations = list(product(model_names, epochs_list, learning_rates, weight_decays))

    for model_name, epochs, learning_rate, weight_decay in combinations:
        json_data = json.dumps({
            "model_name": model_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay
        })
        queue.put(json_data)
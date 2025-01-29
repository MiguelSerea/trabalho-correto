from itertools import product
import time
import logging
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.shortcuts import render
import threading
from joblib import Parallel, delayed
import torch
from .cnn import CNN, define_transforms, read_images
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def index_view(request):
    """Render the main template."""
    return render(request, 'index.html')

@csrf_exempt
def start_training(request):
    """Start the training process when the button is clicked."""
    if request.method == 'POST':
        threading.Thread(target=run_training_process).start()
        return JsonResponse({"message": "Training started"}, status=200)
    return JsonResponse({"error": "Method not allowed"}, status=405)

def run_training_process():
    """Run the training process in a separate thread."""
    try:
        data_transforms = define_transforms(224, 224)
        train_data, validation_data, test_data = read_images(data_transforms)
        cnn_model = CNN(train_data, validation_data, test_data, batch_size=8)
        
        model_name = "resnet18"
        epochs = 20
        learning_rate = 0.0001
        weight_decay = 0.0001
        replications = 10

        save_directory = 'v1/modelos'
        os.makedirs(save_directory, exist_ok=True)

        average_accuracy, best_replication = cnn_model.create_and_train_cnn(
            model_name, epochs, learning_rate, weight_decay, replications
        )
        logging.info(f"Training completed. Average accuracy: {average_accuracy}, Best replication: {best_replication}")

        # Save the best model
        torch.save(cnn_model.state_dict(), os.path.join(save_directory, f"best_model.pt"))
    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)

@csrf_exempt
def centralized_multiprocess_view(request):
    """Start the centralized multiprocess training (saves to v2/modelos)."""
    if request.method == 'POST':
        threading.Thread(target=run_centralized_multiprocess).start()
        return JsonResponse({"message": "Centralized multiprocess training started."}, status=200)
    return JsonResponse({"error": "Method not allowed"}, status=405)

def run_centralized_multiprocess():
    """Run the centralized multiprocess training (saves to v2/modelos)."""
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
    """Start the centralized uniprocess training (saves to v1/modelos)."""
    if request.method == 'POST':
        threading.Thread(target=run_centralized_uniprocess).start()
        return JsonResponse({"message": "Centralized uniprocess training started."}, status=200)
    return JsonResponse({"error": "Method not allowed"}, status=405)

def run_centralized_uniprocess():
    """Run the centralized uniprocess training (saves to v1/modelos)."""
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

import torch
import time
import logging

def process_combination(args):
    """Process a combination of parameters for training."""
    model_name, epochs, learning_rate, weight_decay, replications, cnn = args

    # Criar modelo da CNN
    model = cnn.create_model(model_name)

    # Criar função de perda
    criterion = torch.nn.CrossEntropyLoss()

    # Definir otimizador
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    for replication in range(replications):
        # Corrigir chamada de train_model com os argumentos corretos
        cnn.train_model(
            model=model,
            train_loader=cnn.train_loader,  # Certifique-se de que esse atributo existe
            optimizer=optimizer,
            criterion=criterion,
            model_name=model_name,
            num_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            replication=replication
        )

        start_time = time.time()
        average_accuracy, better_replication = cnn.create_and_train_cnn(
            model_name, epochs, learning_rate, weight_decay, replications
        )
        end_time = time.time()

        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        milliseconds = int((duration * 1000) % 1000)

        result_text = (
            f"Combination:\n"
            f"\t{model_name} - {epochs} - {learning_rate} - {weight_decay}\n"
            f"\tAverage Accuracy: {average_accuracy}\n"
            f"\tBetter Replication: {better_replication}\n"
            f"\tExecution Time: {hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}\n\n"
        )

        logging.info(result_text)
        return result_text

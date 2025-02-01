import os
import json
import logging
from itertools import product
import shutil
import threading
import time
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from threading import Lock
from multiprocessing import Queue
from joblib import Parallel, delayed
import requests
import torch
from interface.utils import clean_models_directory
from interface.cnn import CNN, define_transforms, read_images



# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Conjunto para armazenar workers registrados
registered_workers = set()

def index_view(request):
    """Renderiza o template principal."""
    return render(request, 'index.html')

# ==================================================
# Treinamento Uniprocessador
# ==================================================

@csrf_exempt
def start_training(request):
    """Inicia o treinamento em um único processo."""
    if request.method == 'POST':
        clean_models_directory('models')
        threading.Thread(target=run_training_process).start()
        return JsonResponse({"message": "Training started"}, status=200)
    return JsonResponse({"error": "Method not allowed"}, status=405)

def run_training_process():
    """Executa o processo de treinamento em um único processo."""
    try:
        # Preparar dados e configurar modelos
        data_transforms = define_transforms(224, 224)
        train_data, validation_data, test_data = read_images(data_transforms)
        cnn_model = CNN(train_data, validation_data, test_data, batch_size=8)

        replications = 10
        model_names = ['alexnet', 'mobilenet_v3_large',
                   'mobilenet_v3_small', 'resnet18', 'resnet101', 'vgg11', 'vgg19']
        epochs = [10, 20]
        learning_rates = [0.001, 0.0001, 0.00001]
        weight_decays = [0, 0.0001]

        # Gera todas as combinações possíveis
        combinations = list(product(model_names, epochs,
            learning_rates, weight_decays))

        i = 0
        
        with open("results.txt", "a") as file:
        # Looping para percorrer todas as combinações e realizar os experimentos
            for model_name, epochs, learning_rate, weight_decay in combinations:
                start_time = time.time()
                average_accuracy, better_replication = cnn_model.create_and_train_cnn(
                    model_name, epochs, learning_rate, weight_decay, replications)
                end_time = time.time()

                duration = end_time - start_time
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                milliseconds = int((duration * 1000) % 1000)

                result_text = (
                    f"Combination {i}:\n"
                    f"\t{model_name} - {epochs} - {learning_rate} - {weight_decay}\n"
                    f"\tAverage Accuracy: {average_accuracy}\n"
                    f"\tBetter Replication: {better_replication}\n"
                    f"\tExecution Time: {hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}\n\n"
                )
                print(result_text)
                i += 1

                file.write(result_text)

        
        logging.info(f"Training completed. result_text: {result_text}")

    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)

# ==================================================
# Treinamento Centralizado com Multiprocessamento
# ==================================================

@csrf_exempt
def centralized_multiprocess_view(request):
    """Inicia o treinamento centralizado com multiprocessamento."""
    if request.method == 'POST':
        clean_models_directory('models')
        threading.Thread(target=run_centralized_multiprocess).start()
        return JsonResponse({"message": "Centralized multiprocess training started."}, status=200)
    return JsonResponse({"error": "Method not allowed"}, status=405)

def run_centralized_multiprocess():
    """Executa o treinamento usando multiprocessos no servidor central."""
    try:
        combinations_with_cnn, results_file = prepare_training_environment("v2/modelos")
        
        with open(results_file, "a") as file:
            results = Parallel(n_jobs=2, backend="multiprocessing")(
                delayed(process_combination)(args) for args in combinations_with_cnn
            )
            file.writelines(results)
    except Exception as e:
        logging.error(f"Error during centralized multiprocess training: {e}", exc_info=True)

def process_combination(args):
    """Processa uma combinação específica de parâmetros do treinamento."""
    model_name, epochs, learning_rate, weight_decay, replications, cnn = args

    model = cnn.create_model(model_name)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    try:
        start_time = time.time()
        accuracies = []
        for replication in range(replications):
            accuracy = cnn.train_model(
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
            accuracies.append(accuracy)
        
        end_time = time.time()
        average_accuracy = sum(accuracies) / len(accuracies)
        best_replication = accuracies.index(max(accuracies))

        duration = end_time - start_time
        hours, rem = divmod(duration, 3600)
        minutes, seconds = divmod(rem, 60)
        milliseconds = int(duration * 1000 % 1000)

        result_text = (
            f"Combination:\n"
            f"\tModel: {model_name} - Epochs: {epochs} - LR: {learning_rate} - WD: {weight_decay}\n"
            f"\tAverage Accuracy: {average_accuracy}\n"
            f"\tBest Replication: {best_replication}\n"
            f"\tExecution Time: {int(hours)}:{int(minutes)}:{int(seconds)}:{milliseconds:03}\n\n"
        )
        logging.info(result_text)
        return result_text
    except Exception as err:
        error_msg = f"Error in processing combination for model {model_name}: {err}"
        logging.error(error_msg, exc_info=True)
        return error_msg

# ==================================================
# Treinamento Descentralizado
# ==================================================

# Configuração da fila e do lock
queue = Queue()
queue_lock = Lock()



@csrf_exempt
def register_worker(request):
    """Registra um worker no sistema."""
    # Imprime o tipo de método da requisição

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print(data)
            machine_id = data.get('machine_id')
            ip_address = data.get('ip_address')
            port = data.get('port')
            status = data.get('status')
            num_cores = data.get('num_cores')
        
            if machine_id and ip_address:
                with queue_lock:
                    registered_workers.add((machine_id, ip_address))
                return JsonResponse({"message": f"Worker {machine_id} registered successfully."}, status=200)
            else:
                return JsonResponse({"error": "error"}, status=400)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format."}, status=400)
    
    return JsonResponse({"error": "Method not allowed"}, status=405)

# Função para enviar tarefas para workers
def send_tasks_to_workers():
    """Envia tarefas da fila para os workers registrados."""
    with queue_lock:
        if registered_workers and not queue.empty():
            while not queue.empty() and registered_workers:
                task_data = queue.get()
                worker_id, worker_url = registered_workers.pop()
                send_task_to_worker(worker_id, worker_url, task_data)
                registered_workers.add((worker_id, worker_url))
    threading.Timer(5.0, send_tasks_to_workers).start()

# Função para enviar uma tarefa para um worker específico
def send_task_to_worker(worker_id, worker_url, task_data):
    """Envia uma tarefa para um worker específico."""
    try:
        response = requests.post(f"{worker_url}/receive-task/", json=json.loads(task_data))
        if response.status_code == 200:
            logging.info(f"Task sent to {worker_id} successfully.")
        else:
            logging.error(f"Failed to send task to {worker_id}. Status Code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error communicating with {worker_id}: {e}")

# Função para popular a fila com combinações
@csrf_exempt
def combine(request):
    if request.method == 'POST':
        # Parâmetros para combinar
        replications = 10
        model_names = ['alexnet', 'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet101', 'vgg11', 'vgg19']
        epochs = [10, 20]
        learning_rates = [0.001, 0.0001, 0.00001]
        weight_decays = [0, 0.0001]

        # Gerar todas as combinações possíveis de hiperparâmetros
        combinations = list(product(model_names, epochs, learning_rates, weight_decays))

        # Adicionar cada combinação na fila após serializar para JSON
        for model_name, epoch, learning_rate, weight_decay in combinations:
            json_data = json.dumps({
                "model_name": model_name,
                "epochs": epoch,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay
            }, indent=4)
            queue.put(json_data)

        return JsonResponse({'message': 'Combinações foram enfileiradas com sucesso.'}, status=200)
    else:
        return JsonResponse({'error': 'Método não permitido.'}, status=405)


def processRequest(receivedJson):
    try:
        with queue_lock:
            status = receivedJson.get('status')
            if status in ['ONLINE', 'FINISHED']:
                if status == 'FINISHED':
                    data = receivedJson.get('data')
                    if data:
                        with open("results.txt", "a") as file:
                            file.write(json.dumps(data) + "\n")
                    else:
                        print("Aviso: Nenhum dado encontrado no JSON para salvar.")
                
                elif status == 'ONLINE':
                    num_cores = receivedJson.get('num_cores')
                    if num_cores:
                        combinations = []  # Lista para armazenar as combinações retiradas da fila
                        for _ in range(num_cores):
                            if not queue.empty():
                                combinations.append(queue.get())
                            else:
                                break
                        # Retorna as combinações armazenadas no formato de um objeto
                        return {"data": combinations}
            
            if not queue.empty():
                return queue.get()
            else:
                return None
    except Exception as e:
        print(f"Erro durante processamento: {e}")
        return None

# View para o endpoint POST
@csrf_exempt
def worker_endpoint(request):
    if request.method == 'POST':
        try:
            # Carrega o JSON recebido
            receivedJson = json.loads(request.body)
            
            # Processa a requisição
            response_data = processRequest(receivedJson)
            
            # Retorna a resposta
            if response_data:
                return JsonResponse(response_data, status=200)
            else:
                return JsonResponse({"message": "Nenhuma tarefa disponível na fila."}, status=200)
        
        except json.JSONDecodeError:
            return JsonResponse({"error": "JSON inválido."}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "Método não permitido."}, status=405)





def prepare_training_environment(save_directory):
    """Prepara o ambiente de treinamento e combinações de parâmetros."""
    data_transforms = define_transforms(224, 224)
    train_data, validation_data, test_data = read_images(data_transforms)
    cnn = CNN(train_data, validation_data, test_data, batch_size=8)

    os.makedirs(save_directory, exist_ok=True)

    replications = 10
    model_names = ['alexnet', 'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet101', 'vgg11', 'vgg19']
    epochs = [10, 20]
    learning_rates = [0.001, 0.0001, 0.00001]
    weight_decays = [0, 0.0001]

    combinations = list(product(model_names, epochs, learning_rates, weight_decays))
    return [
        (model_name, epoch, learning_rate, weight_decay, replications, cnn)
        for model_name, epoch, learning_rate, weight_decay in combinations
    ], os.path.join(save_directory, "results.txt")
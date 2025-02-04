import os
import json
import logging
from itertools import product
import shutil

import socket
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
registered_workers = []
task_queue = Queue()
finished_workers = 0
queue = Queue()
queue_lock = Lock()
available_workers = 0


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
        return JsonResponse({"message": "Centralized Uniprocess training started."}, status=200)
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




@csrf_exempt
def register_worker(request):
    """Registra um worker no sistema."""
    global available_workers  # Declare aqui que está utilizando a variável global

    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            machine_id = data.get('machine_id')
            ip_address = data.get('ip_address')
            status = data.get('status')
            num_cores = data.get('num_cores')
        
            if machine_id and ip_address:
                with queue_lock:
                    registered_workers.append((machine_id, ip_address))
                    available_workers += 1  # Controle preciso do número de workers
                    task_queue.put((machine_id, ip_address, status, num_cores,))
                return JsonResponse({"message": f"Worker {machine_id} registered successfully."}, status=200)
            else:
                return JsonResponse({"error": "machine_id or ip_address missing."}, status=400)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format."}, status=400)
    
    return JsonResponse({"error": "Method not allowed"}, status=405)

# Função para popular a fila com combinações
@csrf_exempt
def combine(request):
    if request.method == 'POST':
        # Parâmetros para combinar
        replications = [1]  # Encapsule o valor inteiro em uma lista
        model_names = ['alexnet', 'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet101', 'vgg11', 'vgg19']
        epochs = [10, 20]
        learning_rates = [0.001, 0.0001, 0.00001]
        weight_decays = [0, 0.0001]

        # Gerar todas as combinações possíveis de hiperparâmetros
        combinations = list(product(replications, model_names, epochs, learning_rates, weight_decays))

        # Adicionar cada combinação na fila após serializar para JSON
        for replication, model_name, epoch, learning_rate, weight_decay in combinations:
            json_data = json.dumps({
                "replications": replication,
                "model_name": model_name,
                "epochs": epoch,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay
            }, indent=4)
            queue.put(json_data)
            print(json_data)
            
        return JsonResponse({'message': 'Combinações foram enfileiradas com sucesso.'}, status=200)
    else:
        return JsonResponse({'error': 'Método não permitido.'}, status=405)


@csrf_exempt
def decentralized_multiprocess_view(request):
    """Inicia o treinamento descentralizado via multiprocessamento."""
    
    if request.method == 'POST':
        threading.Thread(target=start_socket_server).start()

        # Manejamento de threads com base no número de workers
        threads = []
      
        # Use a condição para garantir proteção contra corrida
        with queue_lock:
            workers_to_use = min(len(registered_workers), available_workers)
        
        for _ in range(workers_to_use):
            thread = threading.Thread(target=processRequest)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
        
        return JsonResponse({"message": "Decentralized multiprocess training completed."}, status=200)

    return JsonResponse({"error": "Method not allowed"}, status=405)



        
def processRequest():
    """Processa uma requisição de worker (descentralizada)."""
    timeout = time.time() + 60  # Timeout de 60 segundos
    
    while True:
        if time.time() > timeout:
            logging.warning("Timeout reached in processRequest")
            break

        with queue_lock:
            if task_queue.empty():
                break

            received_tuple = task_queue.get()

        keys = ['machine_id', 'ip_address', 'status']
        receivedJson = dict(zip(keys, received_tuple))
        status = receivedJson.get('status')

        if status == 'ONLINE':
            # Calcula o número de combinações por worker
            total_combinations = queue.qsize()  # Número total de combinações na fila
            num_workers = len(registered_workers)  # Número de workers registrados
            combinations_per_worker = total_combinations // num_workers  # Divisão igualitária

            combinations = []
            with queue_lock:
                for _ in range(combinations_per_worker):
                    if not queue.empty():
                        combinations.append(queue.get())  # Adiciona a combinação à lista
                    else:
                        break

            # Prepara o JSON a ser enviado
            json_to_send = {"data": combinations}

            # Envia o JSON para o worker
            try:
                send_json(json_to_send, receivedJson.get('ip_address') + 'receive_task/')
                logging.info(f"Sent {len(combinations)} combinations to {receivedJson.get('ip_address')}")
            except Exception as e:
                logging.error(f"Failed to send JSON to {receivedJson.get('ip_address')}: {e}")

        elif status == 'FINISHED':
            data = receivedJson.get('data')
            if data and isinstance(data, (list, dict)):
                with open("results.txt", "a") as file:
                    file.write(json.dumps(data) + "\n")
                logging.info("Data saved successfully")
            
            

@csrf_exempt          
def send_json(json_data, destination_url):
    """
    Envia dados JSON para a URL de destino especificada, usando o método HTTP POST.
    """
    headers = {'Content-Type': 'application/json'}

    if isinstance(json_data, dict) and 'data' in json_data:
        json_data['data'] = [json.loads(item) if isinstance(item, str) else item for item in json_data['data']]

    # Converte o dicionário em JSON string formatada para depuração correta
    json_string = json.dumps(json_data, indent=4)
    print(json_string)  # Para verificar que está correto

    try:
        response = requests.post(destination_url, json=json_data, headers=headers)
        response.raise_for_status()
        print(f"JSON enviado com sucesso para {destination_url}")
    except requests.exceptions.RequestException as e:
        print(f"Erro ao enviar JSON para {destination_url}: {e}")
        
@csrf_exempt
def record_results(request):
    if request.method == 'POST':
        try:
            # Decodifica o JSON do corpo da requisição
            data = json.loads(request.body)

            # Serialize e escreva o JSON no arquivo `results.txt`
            with open('results.txt', 'a') as file:
                file.write(json.dumps(data, indent=4))
                file.write('\n')

            return JsonResponse({'message': 'Resultados gravados com sucesso.'}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON inválido.'}, status=400)

        except Exception as e:
            return JsonResponse({'error': f'Erro ao gravar resultados: {str(e)}'}, status=500)

    # Método não permitido
    return JsonResponse({'error': 'Método não permitido.'}, status=405)
    
        


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
    
    
# Configurações do socket do servidor
SERVER_HOST = '0.0.0.0'  # Escuta em todas as interfaces
SERVER_PORT = 5000       # Porta para conexão dos workers
BUFFER_SIZE = 4096       # Tamanho do buffer para receber dados

def start_socket_server():
    """Inicia o servidor socket para comunicação com os workers."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(5)  # Aceita até 5 conexões simultâneas
    logging.info(f"Socket server listening on {SERVER_HOST}:{SERVER_PORT}")

    while True:
        client_socket, client_address = server_socket.accept()
        logging.info(f"New connection from {client_address}")
        threading.Thread(target=handle_worker, args=(client_socket,)).start()

def handle_worker(client_socket):
    """Lida com a comunicação com um worker conectado."""
    try:
        while True:
            # Envia uma tarefa para o worker
            if not task_queue.empty():
                task = task_queue.get()
                client_socket.send(json.dumps(task).encode('utf-8'))
                logging.info(f"Sent task to worker: {task}")

            # Recebe o resultado do worker
            result = client_socket.recv(BUFFER_SIZE).decode('utf-8')
            if result:
                logging.info(f"Received result from worker: {result}")
                with open("results.txt", "a") as file:
                    file.write(result + "\n")
            else:
                break
    except Exception as e:
        logging.error(f"Error handling worker: {e}")
    finally:
        client_socket.close()
        logging.info("Worker disconnected")

# Inicia o servidor socket em uma thread separada
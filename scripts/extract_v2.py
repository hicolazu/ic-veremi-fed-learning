import os
import json
import csv
import re
from math import sqrt, atan2

# Diretório configurado para o ambiente atual
base_dir = os.path.dirname(os.path.abspath(__file__))

# Nome do arquivo de saída
output_file = 'VeReMi_v2.csv'

# Função para carregar o GroundTruthJSONlog.json e indexar por messageID
def load_ground_truth(filepath):
    ground_truth = {}
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            ground_truth[entry['messageID']] = entry
    return ground_truth

# Função para calcular a distância euclidiana
def calculate_distance(p, q):
    return sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2)

# Função para calcular o ângulo de chegada
def calculate_aoa(p, q):
    return atan2(q[1] - p[1], q[0] - p[0])

# Função para processar os dados e criar preditores
def process_simulation_with_features(simulation_dir, ground_truth, writer):
    messages_by_sender = {}
    last_type_2_message = None  # Última mensagem do tipo 2 recebida
    type_2_messages_by_message_id = {}  # Mapeia messageID para última mensagem do tipo 2

    for filename in os.listdir(simulation_dir):
        if filename.startswith("JSONlog-") and filename.endswith(".json"):
            with open(os.path.join(simulation_dir, filename), 'r') as f:
                for line in f:
                    message = json.loads(line)

                    if message['type'] == 2:
                        # Atualiza a última mensagem do tipo 2
                        last_type_2_message = message
                    elif message['type'] == 3:
                        # Salva a última mensagem do tipo 2 no mapeamento pelo messageID
                        if last_type_2_message:
                            type_2_messages_by_message_id[message['messageID']] = last_type_2_message

                        sender = message['sender']
                        attacker_type = ground_truth.get(message['messageID'], {}).get("attackerType", "Unknown")
                        if sender not in messages_by_sender:
                            messages_by_sender[sender] = []
                        messages_by_sender[sender].append({
                            "rcvTime": message["rcvTime"],
                            "pos": message["pos"],
                            "pos_noise": message["pos_noise"],
                            "spd": message["spd"],
                            "spd_noise": message["spd_noise"],
                            "messageID": message["messageID"],
                            "RSSI": message["RSSI"],
                            "attackerType": attacker_type
                        })

    # Criar as janelas de 23 mensagens e calcular preditores
    for sender, messages in messages_by_sender.items():
        step = 23
        for i in range(0, len(messages), step):
            window = messages[i:i+step]
            if len(window) < 2:
                continue  # Ignorar janelas com menos de 2 mensagens
            first_message = window[0]
            last_message = window[-1]

            # Recuperar as mensagens do tipo 2 associadas pelo messageID
            first_type_2 = type_2_messages_by_message_id.get(first_message["messageID"])
            last_type_2 = type_2_messages_by_message_id.get(last_message["messageID"])

            if not first_type_2 or not last_type_2:
                continue  # Ignorar se não houver mensagens do tipo 2 correspondentes

            # Calcular preditores
            distance0 = calculate_distance(first_type_2['pos'], first_message['pos'])
            distance1 = calculate_distance(last_type_2['pos'], last_message['pos'])
            aoa0 = calculate_aoa(first_type_2['pos'], first_message['pos'])
            aoa1 = calculate_aoa(last_type_2['pos'], last_message['pos'])
            rssi0 = first_message["RSSI"]
            rssi1 = last_message["RSSI"]

            # Preditor de conformidade (diferença entre posição estimada e reportada)
            delta_time = last_message['rcvTime'] - first_message['rcvTime']
            estimated_position = [
                first_message['pos'][0] + first_message['spd'][0] * delta_time,
                first_message['pos'][1] + first_message['spd'][1] * delta_time,
                first_message['pos'][2] + first_message['spd'][2] * delta_time
            ]
            conformity1 = calculate_distance(estimated_position, last_message['pos'])

            # Criar linha para o CSV
            row = [aoa0, aoa1, rssi0, rssi1, distance0, distance1, conformity1, last_message["attackerType"]]
            writer.writerow(row)

# Função principal para processar simulações
def main():
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['aoa0', 'aoa1', 'RSSI0', 'RSSI1', 'distance0', 'distance1', 'conformity1', 'attackerType']
        writer.writerow(header)

        for root, dirs, files in os.walk(base_dir):
            if 'GroundTruthJSONlog.json' in files:
                ground_truth_file = os.path.join(root, 'GroundTruthJSONlog.json')
                ground_truth = load_ground_truth(ground_truth_file)

                process_simulation_with_features(root, ground_truth, writer)

if __name__ == '__main__':
    main()

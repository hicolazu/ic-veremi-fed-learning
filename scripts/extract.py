import os
import json
import csv
import re

# Diretório configurado hard-coded
base_dir = os.path.dirname(os.path.abspath(__file__))

# Nome do arquivo de saída
output_file = 'VeReMi.csv'

# Função para carregar GroundTruthJSONlog.json e indexá-lo por messageID
def load_ground_truth(filepath):
    ground_truth = {}
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            ground_truth[entry['messageID']] = entry
    return ground_truth

# Função para processar arquivos JSONlog-0-7-A0.json
def process_simulation(simulation_dir, ground_truth):
    result_data = []
    last_type_2_message = None

    for filename in os.listdir(simulation_dir):
        if filename.startswith("JSONlog-") and filename.endswith(".json"):
            # Extrai o identificador do receiver do nome do arquivo
            receiver = int(re.search(r'JSONlog-(\d+)', filename).group(1))

            with open(os.path.join(simulation_dir, filename), 'r') as f:
                for line in f:
                    message = json.loads(line)

                    if message['type'] == 2:
                        # Armazena a última mensagem do tipo 2
                        last_type_2_message = message
                    elif message['type'] == 3 and last_type_2_message:
                        # Busca no ground truth usando o messageID
                        sender_message = ground_truth.get(message['messageID'])

                        if sender_message:
                            # Extrai os campos necessários e cria uma linha para o CSV
                            row = [
                                receiver,
                                message['rcvTime'],
                                last_type_2_message['pos'][0],
                                last_type_2_message['pos'][1],
                                last_type_2_message['noise'][0],
                                last_type_2_message['noise'][1],
                                last_type_2_message['spd'][0],
                                last_type_2_message['spd'][1],
                                last_type_2_message['spd_noise'][0],
                                last_type_2_message['spd_noise'][1],
                                sender_message['pos'][0],
                                sender_message['pos'][1],
                                sender_message['pos_noise'][0],
                                sender_message['pos_noise'][1],
                                sender_message['spd'][0],
                                sender_message['spd'][1],
                                sender_message['spd_noise'][0],
                                sender_message['spd_noise'][1],
                                sender_message['attackerType']
                            ]
                            result_data.append(row)
    return result_data

# Função principal para processar todas as simulações e salvar o CSV
def main():
    all_data = []

    # Percorre cada subdiretório
    for root, dirs, files in os.walk(base_dir):
        if 'GroundTruthJSONlog.json' in files:
            ground_truth_file = os.path.join(root, 'GroundTruthJSONlog.json')
            ground_truth = load_ground_truth(ground_truth_file)

            simulation_data = process_simulation(root, ground_truth)
            all_data.extend(simulation_data)

    # Escreve o arquivo CSV de saída
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [
            'receiver', 'rcvTime', 'pos_rcv_0', 'pos_rcv_1', 'noise_rcv_0', 'noise_rcv_1',
            'spd_rcv_0', 'spd_rcv_1', 'spd_noise_rcv_0', 'spd_noise_rcv_1',
            'pos_snd_0', 'pos_snd_1', 'noise_snd_0', 'noise_snd_1',
            'spd_snd_0', 'spd_snd_1', 'spd_noise_snd_0', 'spd_noise_snd_1', 'attack_type'
        ]
        writer.writerow(header)
        writer.writerows(all_data)

if __name__ == '__main__':
    main()


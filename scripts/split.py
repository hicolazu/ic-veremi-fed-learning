import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv_balanced(input_file, output_files):
    # Lê o arquivo CSV
    data = pd.read_csv(input_file)
    
    # Verifica se a coluna attackerType está presente
    if 'attackerType' not in data.columns:
        raise ValueError("O arquivo CSV deve conter a coluna 'attackerType'")
    
    # Inicializa listas para armazenar os dados de cada arquivo de saída
    csv_parts = [pd.DataFrame(columns=data.columns) for _ in output_files]
    
    # Faz a separação dos dados, garantindo balanceamento para cada classe
    for attacker_type in data['attackerType'].unique():
        subset = data[data['attackerType'] == attacker_type]
        
        # Divide os dados dessa classe em 3 partes balanceadas
        part1, part2 = train_test_split(subset, test_size=2/3, random_state=42)
        part2, part3 = train_test_split(part2, test_size=0.5, random_state=42)
        
        # Adiciona os dados divididos às partes correspondentes
        csv_parts[0] = pd.concat([csv_parts[0], part1])
        csv_parts[1] = pd.concat([csv_parts[1], part2])
        csv_parts[2] = pd.concat([csv_parts[2], part3])
    
    # Salva os arquivos de saída
    for file, part in zip(output_files, csv_parts):
        part.to_csv(file, index=False)
        print(f"Arquivo gerado: {file} com {len(part)} linhas.")

# Configuração
input_csv = "VeReMi_v2.csv"  # Caminho para o arquivo CSV original
output_csvs = ["VeReMi_1.csv", "VeReMi_2.csv", "VeReMi_3.csv"]  # Nomes dos arquivos de saída

# Executa a função de separação
split_csv_balanced(input_csv, output_csvs)

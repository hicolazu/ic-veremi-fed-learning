import os
import requests
from tqdm import tqdm
import tarfile

# URL base para download
base_url = "https://github.com/VeReMi-dataset/VeReMi/releases/download/v1.0/"

# Ler os nomes dos arquivos do arquivo files.txt
files = []
with open("files.txt", "r") as file:
    files = [line.strip() for line in file if line.strip()]

# Diretório de download e extração
download_dir = "downloads"
os.makedirs(download_dir, exist_ok=True)

def download_file(url, output_path):
    """Baixa um arquivo com barra de progresso."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Verifica se houve erro na requisição
        total = int(response.headers.get('content-length', 0))
        with open(output_path, 'wb') as file, tqdm(
            desc=f"Baixando {os.path.basename(output_path)}",
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                bar.update(len(data))
                file.write(data)
    except (requests.RequestException, requests.HTTPError) as e:
        print(f"Erro ao baixar {url}: {e}")
        return False  # Indica que o download falhou
    return True  # Indica que o download foi bem-sucedido

def extract_tgz(file_path, extract_to):
    """Extrai um arquivo .tgz para uma pasta específica."""
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
    except tarfile.TarError as e:
        print(f"Erro ao extrair {file_path}: {e}")

# Processar cada arquivo
for file_name in files:
    file_url = base_url + file_name
    file_path = os.path.join(download_dir, file_name)
    extract_path = os.path.join(download_dir, file_name.replace('.tgz', ''))

    # Download
    print(f"Iniciando download de {file_name}...")
    if download_file(file_url, file_path):
        # Extração
        print(f"Extraindo {file_name} para {extract_path}...")
        os.makedirs(extract_path, exist_ok=True)
        extract_tgz(file_path, extract_path)
    else:
        print(f"Arquivo {file_name} ignorado devido a erro no download.")

print("Processo concluído.")

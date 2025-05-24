# MPII Human Pose Dataset Processor

Este script Python automatiza o processamento do dataset MPII Human Pose, focando na filtragem de imagens por classes de pose específicas (sentado, em pé, andando) e na organização dessas imagens em uma estrutura de diretórios para treinamento de modelos de aprendizado de máquina.
O conjunto de dados MPII Human Pose é um benchmark de última geração para avaliação da estimativa de pose humana articulada. O conjunto de dados inclui cerca de 25 mil imagens contendo mais de 40 mil pessoas com articulações corporais anotadas. As imagens foram coletadas sistematicamente utilizando uma taxonomia estabelecida de atividades humanas cotidianas. No geral, o conjunto de dados abrange 410 atividades humanas e cada imagem é fornecida com um rótulo de atividade. Cada imagem foi extraída de um vídeo do YouTube e fornecida com quadros anteriores e posteriores não anotados.

## Dados:
* Images (12.9 GB)
* Annotations (12.5 MB)

Fonte: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download

## Funcionalidades

1.  **Carregamento de Dados MPII:** Lê o arquivo de anotações `.mat` do MPII.
2.  **Filtragem por Atividade:** Filtra as imagens com base em um conjunto pré-definido de atividades que correspondem às classes de pose "sitting" (sentado), "standing" (em pé) e "walking" (andando/em movimento).
3.  **Mapeamento de Classes:** Mapeia as atividades detalhadas do MPII para as três classes de pose simplificadas.
4.  **Criação de Estrutura de Diretórios:** Cria automaticamente uma estrutura de pastas para organizar os dados processados, divididos por classe e por conjunto (treino, validação, teste).
5.  **Divisão do Dataset:** Divide as imagens filtradas em conjuntos de treinamento (70%), validação (20%) e teste (10%) de forma estratificada (quando possível) para tentar manter a proporção das classes em cada conjunto.
6.  **Cópia de Imagens:** Copia os arquivos de imagem para as pastas correspondentes na estrutura de diretórios processada.
7.  **Geração de Plots:**
    *   Salva um gráfico de barras mostrando a distribuição da quantidade de imagens em cada classe de pose após a filtragem.
    *   Salva um plot com uma imagem de exemplo para cada classe de pose.
    Os plots são salvos na pasta `data/processed_mpii_poses/plots/`.


**Observações Importantes:**
*   O script `filter_images.py` deve estar localizado dentro da pasta `code/`.
*   A pasta `data/` deve estar no mesmo nível da pasta `code/` (ou seja, ambos são subdiretórios diretos de `seu_projeto_raiz/`).
*   O script acessa os dados usando caminhos relativos como `../data`, assumindo que ele é executado a partir do diretório `code/`. Se você executar de `seu_projeto_raiz/` (ex: `python code/filter_images.py`), os caminhos relativos para os dados (`../data`) não funcionarão. **Execute o script de dentro do diretório `code/` ou ajuste o `base_path` no construtor da classe `PoseDatasetProcessor` no script.**

## Pré-requisitos

*   Python 3.7+
*   Bibliotecas Python listadas no arquivo `requirements.txt`.

## Instalação de Dependências

1.  Navegue até o diretório raiz do seu projeto (`seu_projeto_raiz/`) no terminal.
2.  Crie e ative um ambiente virtual (recomendado):
    ```bash
    python -m venv venv
    # No Windows:
    # venv\Scripts\activate
    # No macOS/Linux:
    # source venv/bin/activate
    ```
3.  Instale as dependências usando o arquivo `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```


## Como Usar

1.  **Configuração Inicial:**
    *   Certifique-se de que a estrutura de arquivos do projeto (conforme descrito acima) está correta.
    *   Verifique se a pasta `data/images/` contém todas as imagens `.jpg` do dataset MPII.
    *   Confirme que o arquivo `data/mpii_human_pose_v1_u12_1.mat` está presente.

2.  **Mapeamento de Atividades (Importante!):**
    Abra o script `code/filter_images.py` e localize a variável de classe `CLASS_MAPPING` dentro da classe `PoseDatasetProcessor`:
    ```python
    class PoseDatasetProcessor:
        CLASS_MAPPING = {
            'sitting': [
                'sitting quietly',
                'sitting, talking in person, on the phone, computer, or text messaging, light effort'
                # ADICIONE AQUI OUTRAS ATIVIDADES DO MPII QUE VOCÊ CONSIDERA 'sitting'
            ],
            'standing': [
                'paddle boarding, standing',
                'standing, doing work'
                # ADICIONE AQUI OUTRAS ATIVIDADES DO MPII QUE VOCÊ CONSIDERA 'standing'
            ],
            'walking': [
                'walking, for exercise, with ski poles',
                'skating, ice dancing'
                # ADICIONE AQUI OUTRAS ATIVIDADES DO MPII QUE VOCÊ CONSIDERA 'walking' OU MOVIMENTO
            ]
        }
        # ...
    ```
    Revise e **ajuste este mapeamento** para incluir todas as atividades específicas do dataset MPII que você deseja categorizar em "sitting", "standing" ou "walking". Os nomes das atividades devem ser listados em **letras minúsculas** e corresponder aos nomes encontrados no arquivo `.mat` (o script já normaliza os nomes lidos do `.mat` para minúsculas e remove espaços extras nas pontas).

3.  **Execução do Script:**
    *   Navegue até o diretório `code/` no seu terminal:
        ```bash
        cd seu_projeto_raiz/code
        ```
    *   Execute o script Python:
        ```bash
        python filter_images.py
        ```

## Saída Gerada

Após a execução bem-sucedida, o script realizará as seguintes ações:

1.  **Criará uma pasta `processed_mpii_poses/` dentro do seu diretório `data/`:**
    ```
    seu_projeto_raiz/
    ├── code/
    │   └── filter_images.py
    └── data/
        ├── images/
        ├── mpii_human_pose_v1_u12_1.mat
        └── processed_mpii_poses/  <-- CRIADO PELO SCRIPT
            ├── plots/             <-- CRIADO PELO SCRIPT
            │   ├── class_distribution.png
            │   └── class_examples.png
            ├── sitting/           <-- CRIADO PELO SCRIPT
            │   ├── train/
            │   │   └── (imagens de treino da classe sitting)
            │   ├── val/
            │   │   └── (imagens de validação da classe sitting)
            │   └── test/
            │       └── (imagens de teste da classe sitting)
            ├── standing/          <-- CRIADO PELO SCRIPT
            │   ├── train/
            │   ├── val/
            │   └── test/
            └── walking/           <-- CRIADO PELO SCRIPT
                ├── train/
                ├── val/
                └── test/
    ```
2.  **Imprimirá no console:**
    *   Progresso do carregamento e filtragem.
    *   Progresso da cópia de imagens para os conjuntos de treino, validação e teste.
    *   Resumos da quantidade de imagens em cada conjunto e classe.
    *   Caminhos onde os plots de distribuição e exemplos foram salvos.

## Observações

*   O script foca na classificação e organização das imagens com base na atividade anotada, não utilizando diretamente os keypoints para a filtragem de classes.
*   Se uma imagem contiver múltiplas atividades que mapeiam para diferentes classes de pose desejadas, a política atual (`drop_duplicates(subset=['image_name'], keep='first')`) atribui a imagem à classe correspondente à *primeira* atividade encontrada na anotação que se encaixa no `CLASS_MAPPING`.

# MPII Human Pose Dataset Processor

Este script Python automatiza o processamento do dataset MPII Human Pose, focando na filtragem de imagens por classes de pose específicas (sentado, em pé, andando) e na organização dessas imagens em uma estrutura de diretórios para treinamento de modelos de aprendizado de máquina.

## Funcionalidades

1.  **Carregamento de Dados MPII:** Lê o arquivo de anotações `.mat` do MPII.
2.  **Filtragem por Atividade:** Filtra as imagens com base em um conjunto pré-definido de atividades que correspondem às classes de pose "sitting" (sentado), "standing" (em pé) e "walking" (andando/em movimento).
3.  **Mapeamento de Classes:** Mapeia as atividades detalhadas do MPII para as três classes de pose simplificadas.
4.  **Criação de Estrutura de Diretórios:** Cria automaticamente uma estrutura de pastas para organizar os dados processados, divididos por classe e por conjunto (treino, validação, teste).
5.  **Divisão do Dataset:** Divide as imagens filtradas em conjuntos de treinamento (70%), validação (20%) e teste (10%) de forma estratificada para tentar manter a proporção das classes em cada conjunto.
6.  **Cópia de Imagens:** Copia os arquivos de imagem para as pastas correspondentes na estrutura de diretórios processada.
7.  **Visualização da Distribuição de Classes:** Plota um gráfico de barras mostrando a quantidade de imagens em cada classe de pose após a filtragem.
8.  **Visualização de Exemplos:** Exibe uma imagem de exemplo para cada classe de pose.

## Pré-requisitos

*   Python 3.7+
*   Bibliotecas Python:
    *   `scipy`
    *   `numpy`
    *   `pandas`
    *   `matplotlib`
    *   `scikit-learn`

Você pode instalar as bibliotecas necessárias usando pip:
```bash
pip install scipy numpy pandas matplotlib scikit-learn

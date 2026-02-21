# Desafio Técnico Individual - Ligia (Trilha NLP)

## Descrição do Projeto
Este repositório contém a solução desenvolvida para o Desafio Técnico Individual do processo seletivo da Liga Acadêmica de Inteligência Artificial (Ligia) da Universidade Federal de Pernambuco (UFPE). O projeto integra a trilha de Processamento de Linguagem Natural (PLN).

O objetivo central é a classificação de textos para distinguir conteúdos informativos legítimos de materiais de desinformação (Fake News), utilizando análise de dados textuais não estruturados. A solução desenvolvida prioriza o desempenho preditivo aliado à interpretabilidade, evitando modelos de "caixa-preta" através do uso de modelos lineares explicáveis e técnicas de XAI.

## Estrutura do Repositório e Modularização
Seguindo as orientações da comissão avaliadora para garantir as melhores práticas de Engenharia de Software e modularidade, o pipeline foi dividido em três etapas principais:

* **01_aed_paulo_teodoro.ipynb**: Notebook dedicado à Análise Exploratória de Dados (AED). Foca na compreensão da distribuição das classes, estatísticas de texto e identificação de padrões linguísticos preliminares.
* **02_treinamento_paulo_teodoro.ipynb**: O núcleo do projeto. Contém o pipeline de pré-processamento, vetorização TF-IDF (unigramas e bigramas) e o treinamento da Regressão Logística. Este notebook é responsável por gerar os artefatos de serialização.
* **03_inferencia_paulo_teodoro.ipynb**: Script otimizado para produção. Ele carrega os pesos já treinados e o vetorizador para realizar predições rápidas no conjunto de teste, gerando o arquivo final para o Kaggle sem necessidade de re-treinamento.
* **paulo-teodoro-nlp-ligia-project.ipynb**: Notebook unificado contendo o histórico completo do desenvolvimento.

## Artefatos de Modelo (Pesos e Serialização)
Para garantir a reprodutibilidade e portabilidade da solução, os seguintes arquivos de "pesos" são fornecidos:

* **modelo_fake_news.pkl**: Arquivo serializado via `joblib` que contém os **pesos (coeficientes)** aprendidos pelo modelo de Regressão Logística durante a fase de treinamento. Ele representa o estado final da inteligência do classificador.
* **vectorizer.pkl**: Contém o estado do `TfidfVectorizer`. É obrigatório para a inferência, pois garante que as novas palavras sejam mapeadas para os mesmos índices numéricos utilizados no treino.

## Metodologia Técnica
* **Vetorização**: Extração de unigramas e bigramas (`ngram_range=(1, 2)`) para capturar o contexto de expressões compostas.
* **Classificação**: Regressão Logística com regularização `C=10` e `class_weight='balanced'` para lidar com eventuais desequilíbrios nas classes.
* **Explicabilidade**: Uso da técnica LIME para fornecer transparência sobre quais termos influenciaram cada predição.

## Instruções de Execução
1. Instale as dependências: `pip install -r requirements.txt`.
2. Certifique-se de que os arquivos `train.csv` e `test.csv` estão na raiz do projeto.
3. Execute os notebooks na ordem: `01` -> `02` -> `03`.

**Best Score Alcançado: 0.98560**

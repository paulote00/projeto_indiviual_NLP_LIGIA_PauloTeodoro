# Desafio Técnico Individual - Ligia (Trilha NLP)

## Descrição do Projeto
Este repositório contém a solução desenvolvida para o Desafio Técnico Individual do processo seletivo da Liga Acadêmica de Inteligência Artificial (Ligia) da Universidade Federal de Pernambuco (UFPE). O projeto integra a trilha de Processamento de Linguagem Natural (PLN).

O objetivo central é a classificação de textos para distinguir conteúdos informativos legítimos de materiais de desinformação (Fake News), utilizando análise de dados textuais não estruturados. A solução desenvolvida prioriza o desempenho preditivo aliado à interpretabilidade, evitando modelos de caixa-preta.

## Estrutura do Repositório
A organização do código atende aos requisitos de clareza, modularidade e reprodutibilidade exigidos pela comissão avaliadora:

* `paulo-teodoro-nlp-ligia-project.ipynb`: Notebook principal contendo o pipeline completo de execução. Ele engloba a preparação dos dados, vetorização, treinamento do modelo, validação interna e inferência.
* `requirements.txt`: Arquivo contendo a lista exata de dependências e bibliotecas necessárias para a execução do projeto de ponta a ponta.
* `modelo_fake_news.pkl`: Artefato do modelo final treinado e serializado, utilizado para a geração das previsões.

## Metodologia e Modelagem
A abordagem técnica foi definida considerando restrições de tempo e recursos computacionais, garantindo alta eficiência:
* Vetorização: Transformação dos textos em representações numéricas utilizando `TfidfVectorizer`, processando a concatenação das colunas de título e corpo da notícia com extração de unigramas e bigramas (`ngram_range=(1, 2)`) para maximizar o contexto semântico.
* Classificação: Treinamento de um algoritmo de Regressão Logística parametrizado com ajuste de regularização (`C=10`), balanceamento de classes (`class_weight='balanced'`) e semente fixa (`random_state=42`) para mitigar vieses e garantir integridade científica.
* Avaliação: O modelo foi otimizado e avaliado com base na métrica F1-Score (Macro Average), conforme diretriz da competição.
* Interpretabilidade (XAI): Aplicação da técnica LIME (Local Interpretable Model-agnostic Explanations) para explicar os fatores e padrões linguísticos que influenciam as decisões do modelo, justificando o seu comportamento.

## Instruções de Execução e Reprodutibilidade
Para reproduzir os resultados submetidos ao Kaggle e gerar as análises de interpretabilidade, siga as etapas abaixo:

1. Instalação das Dependências:
Certifique-se de ter o Python instalado e execute o seguinte comando no terminal para instalar os pacotes necessários:
`pip install -r requirements.txt`

2. Download dos Dados:
Acesse a página da competição no Kaggle e faça o download dos arquivos `train.csv` e `test.csv`. Posicione ambos os arquivos no mesmo diretório em que se encontra o arquivo do notebook.

3. Execução do Pipeline:
Como o código foi estruturado em um Jupyter Notebook (.ipynb), você pode reproduzi-lo abrindo o arquivo `paulo-teodoro-nlp-ligia-project.ipynb` em seu ambiente de preferência (Jupyter, VS Code, Google Colab ou Kaggle) e executando as células sequencialmente.

4. Saídas Esperadas:
A execução do script completará as seguintes rotinas automaticamente:
* Treinamento do modelo.
* Exibição das métricas de F1-Score e Matriz de Confusão no console.
* Geração e salvamento dos relatórios visuais de explicabilidade (LIME) em formato `.html`.
* Geração do arquivo `submission.csv` contendo as previsões formatadas para submissão na plataforma Kaggle.
* Serialização e atualização do arquivo `modelo_fake_news.pkl`.

# Previsão de Cancelamento de Reservas — Rede Hoteleira

🏆 Desenvolvido durante o **CDS Hackdays 4** (comunidade Comunidade Data Science), em equipe (**Only Outliers**)

## Contexto

A rede hoteleira espanhola *Costa del Data* enfrentava uma alta taxa de cancelamento de reservas. A diretoria suspeitava de uma mudança no comportamento de cancelamento dos hóspedes após a pandemia, mas ainda não havia dados que confirmassem essa hipótese.

Nossa equipe foi desafiada a desenvolver um modelo de previsão de cancelamentos para apoiar o time de marketing na tomada de decisões mais assertivas.

**Pergunta de negócio:** entre os clientes que fizeram reservas, quais irão cancelar — e o que a empresa pode fazer a respeito?

## Meu papel

Atuei na equipe em todas as etapas do projeto: análise exploratória, formulação e teste de hipóteses de negócio, construção do pipeline de pré-processamento e modelagem, otimização de hiperparâmetros e geração da estratégia de ensemble para a submissão final.

## Dataset

- **Treino**: 72.159 reservas, 15 colunas (incluindo a variável-alvo `Reserva Cancelada`)
- **Teste**: 48.106 reservas, 14 colunas (sem a variável-alvo, usada para a submissão)
- Variáveis incluem: classificação do hotel, antecedência da reserva, número de pernoites, número de hóspedes, regime de alimentação, nacionalidade, forma de reserva, histórico do cliente, tipo de quarto, entre outras.

## Análise Exploratória e Hipóteses de Negócio

Usamos o [Sweetviz](https://github.com/fbdesignpro/sweetviz) para uma primeira leitura automatizada dos dados, e em seguida testamos hipóteses de negócio específicas com visualizações direcionadas.

**Principais achados da EDA:**
- Nacionalidade tinha 1,5% de dados nulos, com 48% dos hóspedes sendo espanhóis
- Diversas variáveis categóricas estavam desbalanceadas (reserva com estacionamento, feita por empresa, feita por agência)
- "Forma de Reserva = Agência" e "Reserva feita por agência de turismo = Sim" **não** eram a mesma coisa — um detalhe fácil de confundir e que afetava o encoding das variáveis

### H1 — Quanto maior a antecedência da reserva, maior a chance de cancelamento

**Confirmada.** A taxa de cancelamento cresce visivelmente conforme aumenta o intervalo entre a reserva e o check-in.

![Taxa de cancelamento vs. antecedência da reserva](images/h1_meses_antecedencia.png)

**Sugestão de negócio:** limitar o período máximo de antecedência para reservas, reduzindo a janela de decisão do cliente.

### H2 — Reservas sem estacionamento cancelam mais

**Confirmada.**

![Taxa de cancelamento vs. estacionamento](images/h2_estacionamento.png)

**Sugestão de negócio:** oferecer estacionamento incluso ou parcerias com estacionamentos próximos como incentivo à manutenção da reserva.

### H3 — Quanto mais pernoites reservadas, menor a chance de cancelamento

**Refutada.** Reservas mais longas não mostraram menor taxa de cancelamento — o padrão não seguiu a intuição inicial da equipe.

**Sugestão de negócio:** revisar a política de cancelamento gratuito para estadias acima de 14 noites.

## Preparação dos Dados e Pipeline

Construímos um `ColumnTransformer` do scikit-learn com pipelines separados para variáveis categóricas e numéricas:

- **Categóricas**: imputação por valor mais frequente + One-Hot Encoding
- **Numéricas**: imputação por mediana + tratamento de outliers (RobustScaler) + padronização (StandardScaler)

Essa estrutura evita vazamento de dados entre treino e teste e mantém o pré-processamento reprodutível.

## Modelagem

- **Algoritmo**: XGBoost Classifier
- **Otimização**: RandomizedSearchCV (20 iterações, validação cruzada 5-fold), buscando o melhor `max_depth`, `gamma`, `min_child_weight`, `colsample_bytree` e `learning_rate`
- **Métrica principal**: F1-score macro, por ser mais robusta a desbalanceamento de classes do que acurácia simples

## Avaliação

O modelo final foi avaliado em dados de teste nunca vistos, com matriz de confusão e métricas de precisão, recall, F1-score e acurácia.

**Precisão**: entre os clientes que o modelo apontou como propensos a cancelar, quantos de fato cancelaram.
**Recall**: entre os clientes que de fato cancelaram, quantos o modelo conseguiu identificar.

## Estratégia de Submissão

Como diversas variações de modelos apresentaram desempenho competitivo entre si, a equipe optou por um **ensemble por votação/média dos 5 melhores modelos** em vez de escolher um único vencedor — reduzindo a variância da previsão final.

| id | modelo 1 | modelo 2 | modelo 3 | modelo 4 | modelo 5 | média | modelo final |
|----|----------|----------|----------|----------|----------|-------|--------------|
| 118345 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9500 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 15492 | 0 | 0 | 0 | 1 | 1 | 0.4 | 0 |

## Próximos Passos

- Investigar o que compõe a variável `id`, que se mostrou correlacionada com outras features — possível vazamento de informação a ser tratado
- Integrar o modelo à etapa de cadastro do cliente na plataforma de reservas, permitindo ação preventiva no momento da reserva (ex: política de cancelamento diferenciada)
- Configurar disparo automático de e-mails para clientes com maior probabilidade de cancelamento
- Monitorar o impacto financeiro das ações tomadas com base no modelo

## Como executar

```bash
pip install -r requirements.txt
jupyter notebook hotel_cancellation_prediction.ipynb
```

## Tecnologias

Python · pandas · scikit-learn · XGBoost · Sweetviz · seaborn · matplotlib

---

*Projeto desenvolvido em equipe durante hackathon da Comunidade Data Science (CDS). Este repositório contém a versão de apresentação do notebook, adaptada para execução fora do ambiente Kaggle original.*

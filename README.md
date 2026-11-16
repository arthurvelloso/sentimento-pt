# Sentimento em português: TF-IDF, BERTimbau e LLM

[![CI](https://github.com/arthurvelloso/sentimento-pt/actions/workflows/ci.yml/badge.svg)](https://github.com/arthurvelloso/sentimento-pt/actions/workflows/ci.yml)

Classificação de sentimento (negativo, neutro, positivo) em avaliações de
produtos em português, com três abordagens comparadas pelo **mesmo avaliador**:
um baseline TF-IDF, o BERTimbau ajustado e um LLM. Comparação em qualidade,
latência e custo.

> **Status:** em construção. Dados, avaliador único, baseline TF-IDF, fine-tuning do
> BERTimbau e a análise de erros do TF-IDF estão prontos. Falta o que depende de GPU:
> a análise de erros e a ablação de dados do BERTimbau, e o LLM (zero/few-shot e QLoRA).

## Resultado

Conjunto de teste (12.618 avaliações), usado uma única vez:

| Modelo | Macro-F1 | Acurácia | Latência p50 / p95 |
| --- | --- | --- | --- |
| TF-IDF + regressão logística | 0,723 | 79,4% | 1,5 ms / 2,3 ms (CPU) |
| BERTimbau ajustado | **0,761** | **86,8%** | 12,1 ms / 16,1 ms (GPU T4) |
| LLM (zero/few-shot e QLoRA) | — | — | — |

F1 por classe do baseline: negativo 0,864 · **neutro 0,442** · positivo 0,863.
A classe neutra é a difícil: 42% dos neutros são previstos como outra classe,
e 17% dos positivos (notas 4-5) são previstos como neutros.

A latência é por exemplo (um texto por chamada, incluindo a vetorização e a
tokenização). **As duas latências não são comparáveis entre si**: o TF-IDF foi medido
numa CPU Apple M5 e o BERTimbau numa GPU Tesla T4 do Colab. Não medi o BERTimbau em
CPU. As métricas de qualidade não dependem da máquina.

O resultado completo (matriz de confusão, grade de validação) está em
[`results/tfidf.json`](results/tfidf.json).

## Dados

[B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01): 132.373
avaliações de clientes de uma loja online brasileira, entre janeiro e maio de
2018. Fixado por commit e verificado por SHA-256.

**Rótulo** a partir da nota: 1-2 → negativo · 3 → neutro · 4-5 → positivo.
Usa-se só o texto da avaliação (`review_text`); a coluna `recommend_to_a_friend`
não entra, para não vazar o rótulo.

**Limpeza, antes do split.** Duplicatas entre treino e teste inflam as métricas.

| Passo | Linhas |
| --- | --- |
| Avaliações no arquivo | 132.373 |
| Sem texto | −3.275 |
| Só pontuação ou emoji (vazio após normalizar) | −106 |
| Mesmo texto com notas diferentes (rótulo ambíguo) | −188 |
| Mesmo texto e mesma nota (fica a primeira cópia) | −2.624 |
| **Avaliações usadas** | **126.180** |

Duas avaliações são "o mesmo texto" se forem iguais após minúsculas, sem
acentos e sem pontuação ("Ótimo produto!" = "otimo produto").

**Split** estratificado, seed 42: treino 100.944 · validação 12.618 · teste 12.618.
Todos têm a mesma distribuição: 25,8% negativo · 12,5% neutro · 61,7% positivo.
O manifesto (`splits/manifest.csv`, só ids de linha) é versionado; os textos não.

## Como reproduzir

Python 3.12+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make test       # testes com dados sintéticos, não baixam nada
make data       # baixa o CSV (49 MB), limpa e confere o manifesto do split
make baseline   # treina, escolhe na validação, avalia no teste, grava results/tfidf.json
make ablation-tfidf  # TF-IDF com 10/25/50/100% do treino + previsões por exemplo (~1 min)
make analysis        # erros por nota e por tamanho, matrizes de confusão, amostra de erros
```

O `make baseline` é lento: a grade tem 8 regressões logísticas sobre ~265 mil
features, o que levou algumas dezenas de minutos no meu Mac. O `make ablation-tfidf`
reaproveita a melhor combinação da grade e leva cerca de 1 minuto.

## Fine-tuning do BERTimbau

[`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
ajustado com `src/train_bert.py`, sobre os mesmos splits e o mesmo avaliador do
baseline. O treino precisa de GPU: neste Mac (MPS) medi cerca de 1,6 s por passo
de 32 exemplos, o que dá, por extrapolação, algo como 4 horas para as 3 épocas.
No Kaggle ou no Colab, use o notebook
[`notebooks/bertimbau_train.ipynb`](notebooks/bertimbau_train.ipynb).

| Escolha | Valor | Por quê |
| --- | --- | --- |
| Comprimento máximo | percentil 95 dos tamanhos em tokens, arredondado a múltiplo de 8 | padding até 512 desperdiça memória; truncar 5% dos textos custa pouco |
| Learning rate | 3e-5 (faixa usual do BERT: 2e-5 a 5e-5) | |
| Épocas / batch | 3 / 32 | |
| Warmup | 10% dos passos, depois decaimento linear | |
| Precisão | fp16 em GPU CUDA | |
| Melhor checkpoint | época com maior **macro-F1 na validação** | mesma métrica do baseline |

**Treino completo** (100.944 exemplos, seed 42), em 23,6 minutos numa Tesla T4:

| Época | Macro-F1 (val) | Acurácia (val) | Loss (val) |
| --- | --- | --- | --- |
| 1 | 0,744 | 86,3% | 0,367 |
| **2** (escolhida) | **0,747** | 86,1% | 0,379 |
| 3 | 0,745 | 85,9% | 0,419 |

O macro-F1 praticamente não muda entre as épocas (diferença de 0,003, provavelmente
dentro do ruído de uma seed), enquanto a loss de validação sobe: o modelo começa a
sobreajustar depois da primeira época. No teste, o checkpoint da época 2 dá macro-F1
de 0,761 e acurácia de 86,8%.

O teste é avaliado uma única vez, com o melhor checkpoint. Um treino parcial
(`--train-subset`) é marcado com `is_subset: true` no JSON, para não ser
confundido com o resultado final. O modelo vai para um repositório **privado** no
Hub a cada época (`--hub-repo`), para uma sessão de GPU que caia não perder o treino.

```bash
pip install -r requirements-train.txt   # e o torch, se não estiver instalado
python -m src.train_bert --hub-repo SEU_USUARIO/bertimbau-sentimento-b2w
# outras seeds:  --seed 1  (grava em results/bertimbau_seed1.json)
```

## Como o baseline foi escolhido

TF-IDF de palavras (1-2 gramas) junto com n-gramas de caracteres (2-5), que pegam
erros de digitação e variações comuns em avaliações. Regressão logística com
`C` e `class_weight` escolhidos na **validação**, por macro-F1:

| C | `class_weight` | Macro-F1 (val) | Acurácia (val) |
| --- | --- | --- | --- |
| **0,25** | **balanced** | **0,717** | 79,2% |
| 0,25 | nenhum | 0,688 | 84,3% |
| 1 | balanced | 0,715 | 80,1% |
| 1 | nenhum | 0,698 | 84,0% |
| 4 | balanced | 0,708 | 80,6% |
| 4 | nenhum | 0,697 | 83,3% |
| 16 | balanced | 0,699 | 80,6% |
| 16 | nenhum | 0,690 | 82,5% |

`class_weight="balanced"` ganha cerca de 2 a 3 pontos de macro-F1 e perde cerca
de 4 pontos de acurácia: troca acertos na classe majoritária por acertos na
neutra. Por isso a métrica principal é macro-F1 e não acurácia.

## Onde o TF-IDF erra

Análise sobre as 12.618 avaliações do teste (`make analysis`), juntando a **nota original**
às previsões. O que o rótulo de 3 classes esconde fica visível por nota:

| Nota | Avaliações | Acerto | Previsto como negativo / neutro / positivo |
| --- | --- | --- | --- |
| 1 | 2.483 | 94,0% | 94% / 5% / 1% |
| 2 | 773 | 70,2% | 70% / 28% / 2% |
| **3 (neutro)** | 1.577 | **57,9%** | 18% / 58% / 24% |
| 4 | 3.173 | 69,0% | 4% / 27% / 69% |
| 5 | 4.612 | 87,7% | 3% / 10% / 88% |

O erro se concentra **nas notas vizinhas ao neutro** (2, 3 e 4), e as notas extremas vão
bem. Isso sugere que parte dos "erros" é fronteira entre classes (uma nota 4 com texto
morno, uma nota 2 com texto brando), e não falta de compreensão. É uma leitura dos números,
ainda não confirmada lendo os textos.

Por tamanho do texto a acurácia varia pouco (82% em textos de 6 a 15 palavras, 76% nos de
mais de 40). Figuras: [`confusao.png`](results/figures/confusao.png) e
[`acuracia_por_nota.png`](results/figures/acuracia_por_nota.png).

## Quanto dado cada modelo precisa

Mesma configuração do baseline (`C=0,25`, `class_weight="balanced"`), treinada com
frações do treino. O ponto de 100% reproduz exatamente o macro-F1 do baseline (0,7230),
o que confere o pipeline.

| Fração | Exemplos | Macro-F1 (validação) | Macro-F1 (teste) |
| --- | --- | --- | --- |
| 10% | 10.094 | 0,696 | 0,700 |
| 25% | 25.236 | 0,706 | 0,711 |
| 50% | 50.472 | 0,710 | 0,713 |
| 100% | 100.944 | 0,717 | 0,723 |

O TF-IDF **satura cedo**: 10 vezes mais dados rendem 2,3 pontos de macro-F1. A curva
lê o teste uma vez por tamanho, para descrever a tendência; nenhuma escolha é feita com ele.

**Ainda por fazer (precisa de GPU):** a mesma curva para o BERTimbau, as previsões por
exemplo do BERTimbau (para a análise de erros dele e a leitura de 40 erros) e o LLM. Tudo
isso roda em [`notebooks/dia11_gpu.ipynb`](notebooks/dia11_gpu.ipynb), que devolve os
arquivos para este repositório.

## Decisões

- **Macro-F1 como métrica principal.** As classes são desbalanceadas (62% positivo),
  então a acurácia premia quem ignora a classe neutra.
- **Um avaliador para todos os modelos** (`src/evaluate.py`): mesmas métricas, mesmo
  teste, mesma forma de medir latência. A comparação não muda por diferença de código.
- **Limpeza antes do split**, com SHA-256 do arquivo e manifesto versionado: o split é
  auditável e reproduzível sem publicar os textos.
- **Teste tocado uma vez**, depois de fixar tudo na validação.
- **BERTimbau sem pesos de classe na loss.** É a receita padrão, mas o baseline ganhou
  com `class_weight="balanced"`, então a comparação favorece um pouco o TF-IDF na classe
  neutra. Uma loss ponderada fica como ablação.

## Limitações

- **A melhor combinação está na borda da grade.** O melhor `C` (0,25) é o menor
  valor testado e o macro-F1 cai a cada aumento de `C`; um `C` menor pode render
  um pouco mais. O baseline pode estar levemente subestimado.
- **Vazamento por produto.** O split é por avaliação, então avaliações do mesmo
  produto aparecem em treino e teste. Em produção, produtos novos tendem a ir pior.
- **Quase-duplicatas "fuzzy" não são tratadas**, só textos idênticos após normalizar.
- **Rótulo por nota é uma aproximação.** Uma nota 4 pode vir com texto neutro, e é
  parte do que a classe neutra tem de difícil.
- **Um domínio e um período:** e-commerce brasileiro, jan-mai de 2018. Não há
  garantia de que generaliza para outros textos.
- **Uma seed e um split.** Não há intervalo de confiança. O fine-tuning de BERT
  varia entre seeds, então a diferença de 3,8 pontos de macro-F1 sobre o TF-IDF
  ainda não tem a variação do BERT medida.
- **A CI não treina o BERT.** Ela roda sem `torch` (leve), então os testes que treinam um
  BERT minúsculo são pulados lá e rodam só localmente (34 testes com torch, 31 sem).
- **O envio ao Hub foi escrito com as assinaturas conferidas, mas não testado ponta a
  ponta** (não havia token na máquina de desenvolvimento). A rodada rápida do notebook
  existe para pegar qualquer problema nisso antes do treino longo.

## Estrutura

```
.
├── configs/default.yaml   # dados, split, TF-IDF e grade
├── src/
│   ├── data.py            # download, rótulo, limpeza, split, manifesto
│   ├── prepare.py         # make data
│   ├── evaluate.py        # avaliador único (qualidade + latência)
│   ├── baseline_tfidf.py  # make baseline
│   ├── ablation_tfidf.py  # curva de tamanho de dados do TF-IDF + previsões
│   ├── train_bert.py      # fine-tuning do BERTimbau (GPU)
│   ├── predict_bert.py    # previsões por exemplo do BERTimbau (GPU)
│   ├── ablation_bert.py   # agrega a curva de dados do BERTimbau
│   ├── llm_classify.py    # LLM em zero/few-shot (GPU)
│   ├── predictions.py     # formato das previsões por exemplo (sem texto)
│   └── analysis.py        # análise de erros e comparação entre modelos
├── splits/                # manifest.csv (ids por split) e meta.json (contagens)
├── results/               # métricas (.json), previsões (preds_*.csv) e figuras
├── tests/
├── notebooks/bertimbau_train.ipynb  # fine-tuning no Kaggle ou no Colab
├── notebooks/dia11_gpu.ipynb        # previsões, ablação e LLM na GPU
└── requirements-train.txt # transformers e accelerate (fora da CI)
```

## Licença

Código sob licença MIT ([LICENSE](LICENSE)).

Os dados são do **B2W-Reviews01**, licenciado sob
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(uso não comercial, com atribuição à B2W Digital e compartilhamento pela mesma
licença). O card do dataset no Hugging Face indica CC BY 4.0, o que diverge do
README original; aqui vale a mais restritiva. Modelos treinados nesses dados devem
herdar essa restrição. Referência: Real, Oshiro e Mafra, *B2W-Reviews01: an open
product reviews corpus*, STIL 2019.

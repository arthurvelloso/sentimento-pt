# Sentimento em português: TF-IDF, BERTimbau e LLM

[![CI](https://github.com/arthurvelloso/sentimento-pt/actions/workflows/ci.yml/badge.svg)](https://github.com/arthurvelloso/sentimento-pt/actions/workflows/ci.yml)

Classificação de sentimento (negativo, neutro, positivo) em avaliações de
produtos em português, com três abordagens comparadas pelo **mesmo avaliador**:
um baseline TF-IDF, o BERTimbau ajustado e um LLM. Comparação em qualidade,
latência e custo.

> **Status:** em construção. Dados, avaliador único e baseline TF-IDF estão
> prontos. BERTimbau e LLM entram nos próximos passos e vão na mesma tabela.

## Resultado

Conjunto de teste (12.618 avaliações), usado uma única vez:

| Modelo | Macro-F1 | Acurácia | Latência p50 / p95 |
| --- | --- | --- | --- |
| TF-IDF + regressão logística | **0,723** | 79,4% | 1,5 ms / 2,3 ms |
| BERTimbau ajustado | — | — | — |
| LLM (zero/few-shot e QLoRA) | — | — | — |

F1 por classe do baseline: negativo 0,864 · **neutro 0,442** · positivo 0,863.
A classe neutra é a difícil: 42% dos neutros são previstos como outra classe,
e 17% dos positivos (notas 4-5) são previstos como neutros.

A latência é por exemplo (um texto por chamada, incluindo a vetorização), em CPU
de um Apple M5. Ela depende da máquina; as métricas de qualidade não.

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
```

O `make baseline` é lento: a grade tem 8 regressões logísticas sobre ~265 mil
features, o que levou algumas dezenas de minutos no meu Mac.

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

## Decisões

- **Macro-F1 como métrica principal.** As classes são desbalanceadas (62% positivo),
  então a acurácia premia quem ignora a classe neutra.
- **Um avaliador para todos os modelos** (`src/evaluate.py`): mesmas métricas, mesmo
  teste, mesma forma de medir latência. A comparação não muda por diferença de código.
- **Limpeza antes do split**, com SHA-256 do arquivo e manifesto versionado: o split é
  auditável e reproduzível sem publicar os textos.
- **Teste tocado uma vez**, depois de fixar tudo na validação.

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
- **Uma seed e um split.** Não há intervalo de confiança.

## Estrutura

```
.
├── configs/default.yaml   # dados, split, TF-IDF e grade
├── src/
│   ├── data.py            # download, rótulo, limpeza, split, manifesto
│   ├── prepare.py         # make data
│   ├── evaluate.py        # avaliador único (qualidade + latência)
│   └── baseline_tfidf.py  # make baseline
├── splits/                # manifest.csv (ids por split) e meta.json (contagens)
├── results/tfidf.json     # métricas de validação e teste
├── tests/
└── notebooks/
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

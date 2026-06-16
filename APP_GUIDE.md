# App Feedback Monitor — Guia da Aplicação

Uma plataforma para analisar automaticamente as avaliações de aplicações móveis da Google Play Store, classificando o sentimento dos utilizadores e identificando os temas mais mencionados. Permite comparar dois métodos de análise diferentes lado a lado.

> **Detalhes técnicos (arquitetura, BD, algoritmos):** ver [technical-guide.md](technical-guide.md).

---

## O que faz esta aplicação?

1. **Recolhe avaliações** da Google Play Store para qualquer aplicação
2. **Classifica automaticamente** o sentimento (positivo, negativo, neutro) e os temas de cada avaliação
3. **Apresenta um dashboard interativo** com gráficos, wordcloud e filtros
4. **Compara dois métodos** de análise diferentes: LLM (inteligência artificial generativa) vs NLP (modelo BERT + tópicos LDA)
5. **Agrupa visualmente** as avaliações por similaridade semântica com embeddings BERT + UMAP + KMeans
6. **Permite lançar todas as pipelines** (scrape, classificação, embeddings) diretamente do browser, sem usar o terminal
7. **Permite avaliar a qualidade** das classificações com um conjunto de etiquetas manuais

---

## Como começar

### 1. Instalar a aplicação

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
cp .env.example .env          # copiar e editar as configurações
```

### 2. Configurar as credenciais (ficheiro `.env`)

Abrir o ficheiro `.env` e preencher conforme necessário:

| Variável | Para que serve |
|----------|---------------|
| `LLM_PROVIDER` | `openai` (GPT na nuvem) ou `ollama` (modelo local, gratuito) |
| `OPENAI_API_KEY` | Chave da API OpenAI — obrigatória se `LLM_PROVIDER=openai` |
| `OPENAI_MODEL` | Modelo OpenAI (ex.: `gpt-4o-mini`) |
| `OLLAMA_BASE_URL` | URL do servidor Ollama (por omissão `http://localhost:11434`) |
| `OLLAMA_MODEL` | Modelo Ollama (ex.: `llama3`) |
| `DEFAULT_APP_ID` | ID da aplicação a analisar por omissão (ex: `com.whatsapp`) |
| `SCRAPE_LANG` | Idioma das avaliações a recolher (ex: `pt` para português) |
| `SCRAPE_COUNTRY` | País da loja a recolher (ex: `pt` para Portugal) |

> Só precisa de **uma** das opções LLM abaixo (GPT **ou** Ollama), não ambas.

### 3. Iniciar o GPT (OpenAI) ou o Ollama

#### Opção A — GPT (OpenAI)

1. Criar uma chave em [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
2. No `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-a-sua-chave
OPENAI_MODEL=gpt-4o-mini
```

3. Testar:

```bash
python -m scripts.classify --app-id com.whatsapp --limit 5
```

4. No dashboard: **⚙️ Pipeline Control → 🤖 LLM Classify** → Provider `(env default)` ou **openai**.

O uso da API é pago por token. Use `--limit` para testes.

#### Opção B — Ollama (local, gratuito)

1. Instalar [Ollama](https://ollama.com).
2. Arrancar o servidor:
   - **macOS / Windows:** abrir a app Ollama, ou no terminal: `ollama serve`
   - **Linux:** `ollama serve` (muitas instalações já arrancam o serviço automaticamente)
3. Descarregar o modelo (só na primeira vez):

```bash
ollama pull llama3
```

4. Confirmar que responde:

```bash
curl http://localhost:11434/api/tags
```

5. No `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

6. Testar:

```bash
python -m scripts.classify --app-id com.whatsapp --provider ollama --limit 5
```

7. No dashboard: **⚙️ Pipeline Control → 🤖 LLM Classify** → Provider **ollama**.

O Ollama tem de estar a correr durante toda a classificação. A primeira execução pode demorar enquanto o modelo carrega na RAM.

#### Opção C — IAEDU (agent-chat da escola)

Para a API **api.iaedu.pt** (multipart + streaming):

```env
LLM_PROVIDER=iaedu
IAEDU_API_KEY=sk-usr-...
IAEDU_CHANNEL_ID=cmpyxvcx42ltml1012mfhvkkp
IAEDU_ENDPOINT=https://api.iaedu.pt/agent-chat/api/v1/agent/SEU_AGENT_ID/stream
```

Testar:

```bash
python -m scripts.classify --app-id com.whatsapp --provider iaedu --limit 5
```

No dashboard: **Pipeline Control → LLM Classify → Provider: iaedu**.

> O endpoint e o `channel_id` vêm do painel **Chatbot API Usage** da IAEDU. Cada review usa um `thread_id` novo automaticamente.

#### Mudar de provider

| Objetivo | Como |
|----------|------|
| Provider por omissão | `LLM_PROVIDER` no `.env` |
| Só numa execução (terminal) | `--provider openai`, `ollama` ou `iaedu` |
| Só numa execução (UI) | Dropdown **Provider** em Pipeline Control |

---

## Como usar — passo a passo

> **Pré-requisito LLM:** se for classificar com o método A, configure e teste [GPT ou Ollama](#3-iniciar-o-gpt-openai-ou-o-ollama) antes deste passo.

### Passo 1 — Recolher avaliações

Antes de fazer qualquer análise, é preciso recolher as avaliações da loja:

```bash
python -m scripts.scrape --app-id com.whatsapp --count 500
```

- `--app-id` — o identificador da aplicação na Google Play (aparece no URL da loja)
- `--count` — quantas avaliações recolher

Outros exemplos:
```bash
# Recolher 200 avaliações do Spotify em inglês, loja dos EUA
python -m scripts.scrape --app-id com.spotify.music --count 200 --lang en --country us

# Recolher as avaliações mais relevantes (em vez das mais recentes)
python -m scripts.scrape --app-id com.whatsapp --count 300 --sort most_relevant
```

Avaliações duplicadas são ignoradas automaticamente.

---

### Passo 2 — Classificar as avaliações

Existem **dois métodos independentes** de classificação. Podem ser usados em separado ou ambos para comparação.

#### Método A — LLM (Inteligência Artificial Generativa)

Usa o GPT ou Ollama para ler cada avaliação e classificar o sentimento e temas.

```bash
python -m scripts.classify --app-id com.whatsapp --limit 500
```

- Mais preciso e capaz de entender contexto e ironia
- Requer API key (OpenAI) ou servidor local (Ollama)
- Mais lento e/ou com custo por avaliação

```bash
# Forçar uso do Ollama (local, gratuito)
python -m scripts.classify --provider ollama --limit 100

# Repetir avaliações que falharam anteriormente
python -m scripts.classify --retry-failed
```

#### Método B — NLP (BERT + LDA)

Usa modelos tradicionais de linguagem natural: BERT para sentimento, LDA para descoberta de tópicos.

```bash
python -m scripts.classify_nlp --app-id com.whatsapp
```

- Não requer API key — corre completamente local
- Primeiro uso descarrega o modelo BERT (~440 MB)
- Sentimentos disponíveis: positivo, negativo, neutro (sem "misto")
- Tópicos descobertos automaticamente pelo modelo LDA

```bash
# Classificar com 12 tópicos LDA em vez de 8
python -m scripts.classify_nlp --num-topics 12

# Re-treinar o modelo LDA após adicionar mais dados
python -m scripts.classify_nlp --retrain-lda

# Para avaliações em inglês
python -m scripts.classify_nlp --language english
```

---

### Passo 3 — (Opcional) Calcular embeddings e clusters

Este passo agrupa visualmente as avaliações por temas semelhantes usando BERT + UMAP + KMeans:

```bash
python -m scripts.embed --app-id com.whatsapp --n-clusters 8
```

- Calcula representações vetoriais (embeddings) para cada avaliação
- Reduz a 2 dimensões com UMAP
- Agrupa as avaliações em clusters com KMeans
- Guarda o resultado em `data/embeddings.npz` para visualização no dashboard

> Pode também fazer isto diretamente no browser — ver **Pipeline Control** abaixo.

---

### Passo 4 — Abrir o dashboard

```bash
streamlit run streamlit_app.py
```

Abre automaticamente em [http://localhost:8501](http://localhost:8501)

---

## O Dashboard — guia visual

### Barra lateral (filtros)

| Filtro | O que faz |
|--------|-----------|
| **App** | Selecionar a aplicação a visualizar |
| **Classification method** | Escolher LLM, NLP, ou ambos em modo comparação |
| **Versions** | Filtrar por versão da aplicação |
| **Sentiments** | Mostrar apenas certos sentimentos (no modo NLP, `mixed` é removido automaticamente) |
| **Topics** | Filtrar por temas específicos |
| **Date range** | Período de tempo das avaliações |
| **Clear cache & reload** | Atualizar dados após correr um classificador |

> Todos os filtros começam **totalmente selecionados** — vê tudo por omissão e vai estreitando à medida que remove valores. O filtro de **Versions** aplica-se a todos os gráficos, incluindo os de agregação por versão.

---

### Separadores do dashboard

#### Sentiment (Sentimento)
- **Gráfico circular** com a distribuição geral de sentimentos (positivo / negativo / neutro / misto)
- **Gráfico de barras** com o sentimento dividido por versão da aplicação

Útil para perceber: *"A versão 2.3 teve mais críticas negativas do que as anteriores?"*

---

#### Topics (Temas)
- **Frequência de temas** — quais os temas mais mencionados pelos utilizadores
- **Temas por versão** — como os temas evoluem entre versões
- **Co-ocorrência de temas** — que temas aparecem frequentemente juntos na mesma avaliação
- **Word Cloud** — nuvem de palavras gerada a partir do texto das avaliações filtradas, mostrando visualmente as palavras mais frequentes
- **Tópicos LDA** (apenas no modo NLP) — os tópicos descobertos automaticamente pelo modelo, com as palavras mais associadas a cada um

Útil para perceber: *"Os utilizadores reclamam mais de bugs ou de performance?"*

---

#### Evolution (Evolução)
- **Pontuação média por versão** — como a nota média (1-5 estrelas) evolui ao longo das versões
- **Sentimento semanal** — gráfico de área com a evolução do sentimento ao longo do tempo

Útil para perceber: *"Depois do update de março, o sentimento melhorou?"*

---

#### Comparison (Comparação) — apenas no modo "Both"

Visível apenas quando se seleciona **"Both (comparison)"** na barra lateral. Compara diretamente os dois métodos:

- **Métricas de acordo** — percentagem de reviews em que LLM e NLP chegaram ao mesmo sentimento
- **Gráficos circulares lado a lado** — distribuição de sentimento de cada método
- **Matriz de concordância** (heatmap) — onde os dois métodos concordam e discordam
- **Comparação de temas** — temas identificados por cada método
- **Exemplos de discordância** — reviews concretas onde LLM e NLP chegaram a conclusões diferentes, com justificação do LLM e palavras-chave do LDA

---

#### Score Analysis (Análise de Classificações)
- **Distribuição de estrelas** — quantas reviews têm 1, 2, 3, 4 ou 5 estrelas
- **Sentimento vs Estrelas** (heatmap) — relação entre a classificação em estrelas e o sentimento detectado

Útil para perceber: *"Há reviews de 4 estrelas com sentimento negativo?"*

---

#### Reviews (Avaliações)
- Lista completa das avaliações com filtro de pesquisa por texto
- Cada avaliação expande para mostrar:
  - Texto completo
  - Sentimento e temas identificados
  - Justificação do LLM (quando disponível)
  - Palavras-chave do tópico LDA (quando disponível)
  - Resposta do programador (quando existe)
  - No modo comparação: resultado de **ambos os métodos** para a mesma review

> Mostra no máximo 100 reviews de cada vez. Usar o filtro de pesquisa para refinar.

---

#### Export CSV
Botão no topo do dashboard para exportar todas as reviews filtradas para um ficheiro CSV.

---

## Pipeline Control — lançar pipelines no browser

Acessível no menu lateral do dashboard como **"⚙️ Pipeline Control"**.

Permite executar todas as operações sem sair do browser, organizado em 4 separadores:

| Separador | O que faz |
|-----------|-----------|
| **🕷️ Scrape** | Recolher avaliações da Google Play — escolher app, quantidade, idioma, país e ordenação |
| **🤖 LLM Classify** | Classificar com LLM — escolher provider (OpenAI/Ollama), retry de falhas |
| **🧬 NLP Classify** | Classificar com BERT + LDA — configurar nº de tópicos, idioma, re-treinar LDA |
| **📊 Embeddings** | Calcular embeddings BERT + clusters KMeans — ver visualização UMAP 2-D interativa |

Cada pipeline corre em segundo plano e o log atualiza automaticamente a cada 3 segundos enquanto o pipeline está a correr. Pode também clicar **"🔄 Refresh log"** manualmente. Use **⏹ Stop** para terminar um pipeline a meio.

**Barra lateral do Pipeline Control:**

- **Package name (App ID)** — aplicado a todos os separadores quando se corre um pipeline
- **Max reviews per classify run** — aplica-se **apenas** a LLM Classify e NLP Classify (0 = todos os pendentes). O Scrape não usa este valor — usa **Number of reviews to fetch** dentro do próprio separador. Os Embeddings têm o seu próprio limite dentro do separador.

> O UMAP 2-D mostra cada avaliação como um ponto; ao passar o rato por cima vê o texto da avaliação, a pontuação em estrelas e a versão da app.

---

## Página About & Help

Acessível no menu lateral como **"📖 About & Help"**.

Contém um guia completo da aplicação dentro do próprio browser: quickstart, explicação separador a separador, referência de filtros, taxonomia de temas, formato do gold dataset e perguntas frequentes.

---

## Ferramenta de Etiquetagem Manual

Acessível no menu lateral do dashboard como **"🏷️ Review Labeling"**.

Permite rotular manualmente reviews para criar um conjunto de dados de referência (gold dataset) para avaliação da qualidade das classificações automáticas.

- Mostra as reviews com a classificação do LLM como sugestão
- Permite confirmar ou corrigir o sentimento e os temas manualmente
- Guarda as etiquetas em `data/gold.jsonl`
- Navegação por páginas (10 reviews por página)
- Filtro para ver só classificadas, só não classificadas, ou todas

---

## Avaliar a qualidade das classificações

Depois de ter um conjunto de etiquetas manuais, é possível medir a precisão de cada método:

```bash
# Avaliar ambos os métodos e comparar
python -m scripts.evaluate --gold data/gold.jsonl

# Avaliar apenas o LLM
python -m scripts.evaluate --gold data/gold.jsonl --method llm

# Avaliar apenas o NLP
python -m scripts.evaluate --gold data/gold.jsonl --method nlp

# Guardar gráficos de resultados
python -m scripts.evaluate --gold data/gold.jsonl --save-plots
```

O relatório inclui: precisão global, F1 por classe, matriz de confusão, e comparação lado a lado entre métodos.

---

## Temas identificados pelos classificadores

| Tema | O que representa |
|------|-----------------|
| `performance` | Velocidade, lentidão, consumo de bateria, crashes |
| `ui_ux` | Design, aspeto visual, navegação |
| `bugs` | Erros, problemas técnicos, funcionalidades partidas |
| `features` | Funcionalidades em falta, pedidos, elogios a funcionalidades |
| `pricing` | Custo, subscrições, compras na app, publicidade |
| `privacy_security` | Privacidade, permissões, segurança dos dados |
| `customer_support` | Suporte ao cliente, respostas da equipa |
| `updates` | Efeito de actualizações recentes |
| `usability` | Facilidade de uso, acessibilidade |
| `other` | Temas que não se enquadram nas categorias acima |

---

## Fluxo de trabalho rápido (resumo)

**Via terminal:**
```
1. Recolher dados        →  python -m scripts.scrape --app-id com.whatsapp --count 500
2. Classificar (LLM)     →  python -m scripts.classify --app-id com.whatsapp
3. Classificar (NLP)     →  python -m scripts.classify_nlp --app-id com.whatsapp
4. Embeddings (opcional) →  python -m scripts.embed --app-id com.whatsapp --n-clusters 8
5. Ver o dashboard       →  streamlit run streamlit_app.py
6. (Opcional) Etiquetar  →  dashboard → Review Labeling
7. (Opcional) Avaliar    →  python -m scripts.evaluate --gold data/gold.jsonl
```

**Alternativa — tudo no browser (sem terminal após o primeiro arranque):**
```
1. streamlit run streamlit_app.py
2. Abrir "⚙️ Pipeline Control" na barra lateral
3. Usar os separadores para fazer Scrape → LLM Classify → NLP Classify → Embeddings
4. Voltar ao dashboard principal para ver os resultados
```

---

## Problemas frequentes

| Problema | Solução |
|----------|---------|
| Dashboard vazio | Verificar se correu o scraper e pelo menos um dos classificadores |
| NLP classifica 0 reviews | Correr `python -m scripts.migrate_db` para corrigir o esquema da base de dados |
| Erro de API OpenAI (rate limit) | O sistema tenta automaticamente até 5 vezes com backoff exponencial; se falhar, aguardar e tentar novamente mais tarde |
| Erro de API OpenAI (chave inválida) | Verificar a `OPENAI_API_KEY` no ficheiro `.env` |
| Ollama: connection refused / timeout | Arrancar Ollama (`ollama serve` ou app), confirmar `curl http://localhost:11434/api/tags`, verificar `OLLAMA_BASE_URL` |
| Ollama: model not found | Correr `ollama pull llama3` (ou o modelo definido em `OLLAMA_MODEL`) |
| Dashboard não actualiza após classificar | Clicar em **"Clear cache & reload"** na barra lateral |
| Primeiro uso NLP muito lento | Normal — está a descarregar o modelo BERT (~440 MB) pela primeira vez |
| Embeddings demoram muito | Normal em CPU — ~30–60 s por 500 reviews. Usar `--limit` para testar primeiro com menos dados |
| Pipeline Control mostra log vazio ao arrancar | Normal — aguardar 3 segundos para o primeiro auto-refresh; clicar **"🔄 Refresh log"** se não atualizar |
| Pipeline Control mostra log desatualizado | O log atualiza automaticamente a cada 3 s enquanto o pipeline corre; clicar **"🔄 Refresh log"** para forçar |
| Sentimento `mixed` desapareceu no modo NLP | Comportamento correto — o NLP (BERT) não produz sentimento misto; o filtro remove-o automaticamente |
| Word Cloud não aparece | Verificar que há reviews carregadas e que os filtros não estão demasiado restritivos |

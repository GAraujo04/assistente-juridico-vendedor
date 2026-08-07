# ⚖️ Assistente Jurídico para Vendedores

Aplicação web local, simples, para ajudar vendedores a analisar contratos e
tirar dúvidas antes de fechar uma venda. Faz upload de contratos (PDF ou
DOCX), gera um resumo automático (partes, valores, prazos/multas e riscos) e
oferece um chat que cruza informações de **todos os contratos já salvos**
para responder perguntas (RAG) — sem precisar reenviar ou reabrir o contrato
certo antes de perguntar.

## Como funciona (visão geral)

- **Interface**: [Streamlit](https://streamlit.io/) — um único app Python, sem necessidade de HTML/CSS/JS separados.
- **Leitura do contrato**: `pypdf` (PDF) e `python-docx` (DOCX).
- **RAG**: o texto de cada contrato é dividido em chunks, transformado em
  embeddings **locais** (biblioteca `sentence-transformers`, modelo
  multilíngue) e guardado numa **única coleção vetorial persistente**
  (ChromaDB) compartilhada por todos os contratos — cada trecho carrega de
  qual arquivo veio, então o chat consegue cruzar informações entre
  contratos diferentes.
- **LLM**: a geração do resumo e das respostas do chat usa um provedor
  compatível com a API da OpenAI (Groq, por padrão — veja `.env.example`).

## Estrutura do projeto

```
assistente-juridico-vendedor/
├── app.py           # arquivo principal — interface e orquestração do app
├── rag_utils.py      # extração de texto, chunking, índice vetorial, prompts do LLM
├── requirements.txt
├── .env.example
└── README.md
```

## Onde colocar a API Key

**Não é no `app.py`.** A chave fica em um arquivo `.env` na raiz do projeto
(mesma pasta do `app.py`). O `app.py` só faz `load_dotenv()` e lê a variável
de ambiente — ele nunca deve conter a chave escrita diretamente.

## Passo a passo para rodar localmente

### 1. Criar e ativar um ambiente virtual (recomendado)

```bash
cd assistente-juridico-vendedor
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

> A primeira instalação pode demorar um pouco por causa do `sentence-transformers`
> (baixa um modelo de embeddings de ~470MB na primeira execução do app).

### 3. Configurar a chave da OpenAI

```bash
cp .env.example .env
```

Abra o arquivo `.env` recém-criado e cole sua chave:

```
OPENAI_API_KEY=sk-sua-chave-real-aqui
OPENAI_MODEL=gpt-4o-mini
```

### 4. Rodar a aplicação

```bash
streamlit run app.py
```

O navegador abrirá automaticamente em `http://localhost:8501`.

## Usando o app

1. Na barra lateral, faça upload de um contrato `.pdf` ou `.docx`.
2. Aguarde o processamento (extração e indexação do texto).
3. Se quiser, clique em "Gerar resumo automático do contrato" para ver:
   - Partes envolvidas
   - Valor total e forma de pagamento
   - Cláusulas de rescisão, multas e prazos
   - Riscos e pontos de atenção para o vendedor
4. Use o chat, a qualquer momento (com ou sem contrato aberto na tela), para
   perguntar coisas específicas, por exemplo:
   - "Qual é a multa por atraso no pagamento?"
   - "O cliente pode cancelar sem custo?"
   - "Qual contrato tem o valor de R$ 45.000?"

O chat pesquisa nos trechos de **todos os contratos já salvos**, não só no
que está com o resumo aberto na tela — se a pergunta bater em mais de um
contrato, a resposta indica de qual arquivo veio cada informação. Se a
informação não existir em nenhum contrato salvo, o assistente informa isso
claramente em vez de inventar uma resposta. Perguntas que não dependem de
nenhum contrato (dúvidas jurídicas gerais, técnicas de venda etc.) também
são respondidas normalmente, deixando claro quando a resposta não vem de um
contrato salvo.

## Persistência dos contratos

Todo contrato enviado é salvo em `data/` (criada automaticamente na raiz do
projeto): o arquivo original, o texto extraído, os embeddings (ChromaDB
persistente) e o resumo, se já tiver sido gerado. Isso significa que:

- Reiniciar o app **não apaga** os contratos já processados.
- Na barra lateral, em "📚 Contratos salvos", dá para reabrir qualquer
  contrato enviado antes sem reprocessar (nem gastar chamada de API de novo,
  se o resumo já tinha sido gerado).
- Reenviar o mesmo arquivo (mesmo conteúdo) reaproveita o processamento já
  feito em vez de indexar tudo de novo.
- A pasta `data/` fica de fora do git (`.gitignore`) porque guarda conteúdo
  de contratos reais — não deve ser commitada nem compartilhada sem cuidado.
  Para apagar tudo o que foi salvo, basta remover a pasta `data/`.

## Observações

- PDFs escaneados como imagem (sem texto selecionável) não são suportados,
  pois não há OCR nesta versão simples.
- Nenhum dado do contrato é enviado a servidores além das chamadas à API do
  provedor de LLM (Groq, por padrão) necessárias para gerar o resumo e as
  respostas do chat.

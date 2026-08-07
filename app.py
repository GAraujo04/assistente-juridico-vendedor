"""
app.py
------
Assistente Jurídico para Vendedores — aplicação Streamlit local.

Fluxo:
1. Vendedor faz upload do contrato (.pdf ou .docx).
2. O texto é extraído, dividido em chunks e indexado em um banco vetorial
   único e persistente (ChromaDB) usando embeddings locais
   (SentenceTransformers) — a mesma coleção acumula os trechos de TODOS os
   contratos já enviados, não só o mais recente.
3. Se o vendedor pedir (botão), um resumo automático é gerado via LLM,
   destacando partes, valores, prazos/multas e riscos para o vendedor.
4. O vendedor pode conversar em um chat a qualquer momento — as respostas
   cruzam informações de TODOS os contratos salvos (RAG), além de responder
   dúvidas jurídicas gerais.

A chave da API fica no arquivo `.env` (veja `.env.example`) — nunca neste arquivo.
"""

import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag_utils import (
    ContractStore,
    answer_question,
    compute_contract_id,
    extract_text,
    generate_summary,
    list_stored_contracts,
    load_contract_text,
    load_summary,
    save_contract,
    save_summary,
    split_into_chunks,
    sync_all_contracts_to_store,
)

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")
# Endpoint do provedor de LLM. Por padrão usa a Groq. Para trocar de provedor no
# futuro (ex.: GitHub Models, OpenAI de verdade), basta definir LLM_BASE_URL no
# .env — nenhuma mudança de código é necessária.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", GROQ_BASE_URL)
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # bom suporte a português

st.set_page_config(
    page_title="Assistente Jurídico para Vendedores",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Estilo simples e limpo
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main .block-container { padding-top: 2rem; max-width: 1100px; }
        h1 { font-weight: 700; }
        .stChatMessage { border-radius: 12px; }
        div[data-testid="stMetric"] {
            background: rgba(120, 120, 120, 0.08);
            border-radius: 10px;
            padding: 0.6rem;
        }
        .subtitle { color: gray; margin-top: -0.6rem; margin-bottom: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Recursos pesados: carregados uma única vez por processo
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando modelo de embeddings local (primeira vez pode demorar)...")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Carregando índice de contratos salvos...")
def get_contract_store(_embedding_model) -> ContractStore:
    """Índice único (compartilhado entre todos os usuários/sessões do processo)
    com os trechos de TODOS os contratos salvos — é nele que o chat pesquisa."""
    store = ContractStore(_embedding_model)
    sync_all_contracts_to_store(store)  # indexa contratos salvos antes desta versão
    return store


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=LLM_BASE_URL)


# ---------------------------------------------------------------------------
# Estado da sessão
# ---------------------------------------------------------------------------

defaults = {
    "processed_file_id": None,
    "contract_id": None,
    "full_text": "",
    "summary": None,
    "messages": [],
    "uploader_key": 0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------------------------
# Sidebar: upload do contrato
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📎 Contrato")
    uploaded_file = st.file_uploader(
        "Envie o contrato do cliente",
        type=["pdf", "docx"],
        key=f"file_uploader_{st.session_state.uploader_key}",
    )

    stored_contracts = list_stored_contracts()
    if stored_contracts:
        st.divider()
        st.subheader("📚 Contratos salvos")
        labels = [
            f"{item['filename']}" + (" ✅" if item["has_summary"] else "")
            for item in stored_contracts
        ]
        picked = st.selectbox(
            "Reabrir um contrato já enviado antes",
            options=range(len(stored_contracts)),
            format_func=lambda i: labels[i],
            index=None,
            placeholder="Escolha um contrato...",
        )
        if picked is not None and st.button("Abrir contrato selecionado"):
            chosen = stored_contracts[picked]

            st.session_state.full_text = load_contract_text(chosen["id"])
            st.session_state.summary = load_summary(chosen["id"])
            st.session_state.messages = []
            st.session_state.contract_id = chosen["id"]
            st.session_state.processed_file_id = ("__stored__", chosen["id"])
            # Limpa o campo de upload: sem isso, um arquivo antigo ainda presente
            # nele seria "redetectado" no próximo rerun e sobrescreveria o
            # contrato que acabamos de abrir.
            st.session_state.uploader_key += 1
            st.rerun()

    st.divider()
    st.caption(
        "1. Faça upload do contrato\n\n"
        "2. Peça o resumo automático quando quiser\n\n"
        "3. Tire dúvidas no chat"
    )

    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    if not api_key_present:
        st.warning(
            "⚠️ Nenhuma OPENAI_API_KEY encontrada.\n\n"
            "Crie um arquivo `.env` (baseado no `.env.example`) na pasta do projeto "
            "e adicione sua chave."
        )


st.title("⚖️ Assistente Jurídico para Vendedores")
st.markdown(
    '<p class="subtitle">Analise contratos e tire dúvidas antes de fechar a venda.</p>',
    unsafe_allow_html=True,
)

client = get_openai_client()

# ---------------------------------------------------------------------------
# Processamento do contrato (só reprocessa se o arquivo mudar)
# ---------------------------------------------------------------------------

if uploaded_file is not None:
    file_id = (uploaded_file.name, uploaded_file.size)

    if file_id != st.session_state.processed_file_id:
        with st.spinner("Lendo e processando o contrato..."):
            file_bytes = uploaded_file.getvalue()
            contract_id = compute_contract_id(file_bytes)

            full_text = extract_text(uploaded_file)

            if not full_text.strip():
                st.error("Não foi possível extrair texto deste arquivo. Verifique se o PDF não é uma imagem escaneada.")
                st.stop()

            chunks = split_into_chunks(full_text)

            embedding_model = load_embedding_model()
            store = get_contract_store(embedding_model)
            store.add_chunks(contract_id, uploaded_file.name, chunks)

            save_contract(contract_id, uploaded_file.name, file_bytes, full_text)

            st.session_state.full_text = full_text
            st.session_state.summary = load_summary(contract_id)  # reaproveita se já existir
            st.session_state.messages = []
            st.session_state.contract_id = contract_id
            st.session_state.processed_file_id = file_id

# ---------------------------------------------------------------------------
# Área principal: resumo sob demanda (se houver contrato) + chat (sempre disponível)
# ---------------------------------------------------------------------------

if not st.session_state.full_text:
    st.info(
        "👈 Envie um contrato (.pdf ou .docx) na barra lateral para pedir um resumo. "
        "Você já pode usar o chat abaixo para tirar dúvidas jurídicas gerais."
    )
elif st.session_state.summary is not None:
    st.subheader("📋 Resumo Automático do Contrato")
    with st.container(border=True):
        st.markdown(st.session_state.summary)
else:
    st.success("Contrato carregado. Peça o resumo automático quando quiser, ou já use o chat abaixo.")
    if st.button("📋 Gerar resumo automático do contrato", disabled=client is None):
        with st.spinner("Gerando resumo automático do contrato..."):
            summary = generate_summary(client, OPENAI_MODEL, st.session_state.full_text)
            st.session_state.summary = summary
            if st.session_state.contract_id:
                save_summary(st.session_state.contract_id, summary)
        st.rerun()
    if client is None:
        st.caption("Configure a OPENAI_API_KEY no .env para gerar o resumo.")

st.subheader("💬 Chat")
st.caption(
    "Perguntas são respondidas cruzando informações de TODOS os contratos já salvos "
    "(não só o que está aberto acima); outras perguntas (jurídicas em geral, técnicas "
    "de venda etc.) também são respondidas, deixando claro quando a resposta não vem "
    "de nenhum contrato salvo."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if client is None:
    st.chat_input("Configure a OPENAI_API_KEY no .env para usar o chat.", disabled=True)
else:
    question = st.chat_input("Ex: Qual é a multa por atraso no pagamento? ou qualquer outra dúvida jurídica")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Pesquisando nos contratos salvos..."):
                embedding_model = load_embedding_model()
                store = get_contract_store(embedding_model)
                relevant_chunks = store.query(question, top_k=5)
                answer = answer_question(client, OPENAI_MODEL, question, relevant_chunks)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

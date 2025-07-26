from operator import itemgetter

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI


def create_rag_chain_lcel(
    retriever, chat_model_name="gpt-3.5-turbo", temperature=0.0, history=None
):
    llm = ChatOpenAI(model=chat_model_name, temperature=temperature)

    # Format chat history for prompt
    if history:
        formatted_history = "\n".join([f"User: {q}" for q in history])
    else:
        formatted_history = ""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """আপনি একজন সহায়ক বাংলা ভাষার AI সহকারী। আপনি শুধুমাত্র নিচে দেয়া 'Context' থেকে তথ্য কাজে লাগিয়ে ব্যবহারকারীর প্রশ্নের উত্তর লিখবেন।
                - যদি Context‑এ উত্তর না থাকে, তবে স্পষ্টভাবে লিখবেন: "দুঃখিত, আমার কাছে এটি উল্লেখিত প্রসঙ্গ নেই।"
                - আপনি কখনো নিজে থেকে কিছু উদ্ভাবন করবেন না, শুধু Context‑এর তথ্য উপস্থাপন করবেন।
                - উত্তর সংক্ষিপ্ত ও পরিষ্কার হবে।""",
            ),
            ("placeholder", formatted_history),
            ("human", "Context:\n{context}\n\nপ্রশ্ন:\n{question}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# --- Optional: Wrapper function for simpler invocation if needed ---
# This mimics the old RetrievalQA.invoke output structure if your main.py expects it
def invoke_rag_chain(chain, query_text):
    """
    Invokes the LCEL RAG chain and formats the output similarly to RetrievalQA.

    Args:
        chain (Runnable): The LCEL chain created by create_rag_chain_lcel.
        query_text (str): The user's question.

    Returns:
        dict: A dictionary with 'result' and 'source_documents' keys.
    """
    inputs = {"question": query_text}
    output_dict = chain.invoke(inputs)

    # Extract results
    result_text = output_dict.get("answer", "No answer generated.")
    source_docs = output_dict.get(
        "context", []
    )  # This will be the list of Document objects

    return {"result": result_text, "source_documents": source_docs}

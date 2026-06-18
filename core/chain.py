from typing import Dict, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

import config

# Defining a system prompt 
# {context} = retrieved chunks, {question} = user's question
SYSTEM_PROMPT = """You are DocuMind, an AI assistant that answers question based ONLY on document context below.

Rules:
- Only use the provided context to answer. Never use outside knowledge.
- If the answer id not in the context, say: "I couldn't find that in the uploaded documents."
- Keep answers clear and concise.
- Mention the source file when relevant

context:
{context}
"""

def format_chunks(docs) ->str:
    """Converts the Document chunks into one readable string for the prompt."""
    return "\n\n---\n\n".join(
        f'[doc.metadata.get("sourcefile", "?")]'
        f'[doc.metadata.get("page", "?")]\n{doc.page_content}'
        for doc in docs
    )

class RAGChain:
        def __init__(self, retriever):
            self._retriever = retriever # Stores the retriever on the _instance
            self._history: List = [] # declare self._history as an empty list to store conversations
            self._last_docs: List = [] # declare self._last_docs as an empty list to store recently retrieved document chunks

            self._llm = ChatGroq(
                api_key=config.GROQ_API_KEY,
                model_name=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                )
            
            prompt = ChatPromptTemplate.from_messages([
                ('system', SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name='history'),
                ('human', '{question}'),
            ])

            # LCEL pipeline: retrieve context > fill prompt > call LLM > parse text
            # The | symbol pipes output from one step into next (like Unix pipes)
            self._chain = (
                RunnablePassthrough.assign(
                    context=lambda x: format_chunks (          # supplies a context parameter: a lambda that receives input x and returns formatted context string
                        self._retriever.invoke(x['question'])  # invoke method with the input x['question'] to retrieve relevant document chunks            
                      )
                    )
                | prompt
                | self._llm
                | StrOutputParser()

            )

        def ask(self, question: str) -> Dict:
            # Fetch source docs for citations
            self._last_docs = self._retriever.invoke(question)

            # Keep only last 5 exchanges in memory
            recent_history = self._history[-10:]

            answer = self._chain.invoke({
                 'question': question,
                 'history': recent_history,
            })

            # Save to memory
            self._history.append(HumanMessage(content=question))
            self._history.append(AIMessage(content=answer))

            return {
                 'answer': answer,
                 'sources': self._get_sources()
            }
        
        def _get_sources(self) -> List[Dict]:
            """Deduplicate and return clean source list."""
            seen = set()
            sources = []

            for doc in self._last_docs:
                key = (doc.metadata.get('source_file'), doc.metadata.get('page'))
                if key not in seen:
                    seen.add(key)
                    sources.append({
                         'file': doc.metadata.get('source_file', 'Unknown'),
                         'page': str(doc.metadata.get('page', 'N/A')),
                         'snippet': doc.page_content[:200] + '...',
                    })
            return sources
        
        def clear_memory(self):
            self._history = []


            
             
        



        





            

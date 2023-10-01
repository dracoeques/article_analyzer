from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import load_prompt
from langchain.chains.llm import LLMChain
from langchain.chains import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.callbacks import get_openai_callback
from openai.error import InvalidRequestError, RateLimitError
import tiktoken

# Summarize articles and get the result from OpenAI using Map-Reduce method
def summarize_article(apikey: str, content: str, ):
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    token_count = len(encoding.encode(content))

    llm = ChatOpenAI(temperature=0, openai_api_key=apikey, model = 'gpt-3.5-turbo', max_tokens=624)

    # Create Document object for the text
    docs = [Document(page_content=content)]

    if token_count < 1872:
        # chunk size of template was calculated (3072-1600)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1872, chunk_overlap=0
        )
        split_docs = text_splitter.split_documents(docs)
    else:
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=11088, chunk_overlap=0
        )
        split_docs = text_splitter.split_documents(docs)
        llm = ChatOpenAI(temperature=0, openai_api_key=apikey, model = 'gpt-3.5-turbo-16k', max_tokens=3696)

    # Map
    map_prompt = load_prompt("./prompts/summarize-map.yaml")
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Run chain
    reduce_prompt = load_prompt("./prompts/summarize-reduce.yaml")
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries", 
        # reduce_k_below_max_tokens=True,
    )
    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=50000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    try:
        with get_openai_callback() as cb:
            if len(split_docs) == 1:
                summary = combine_documents_chain.run(split_docs)
                return [summary, cb]
            else:
                summary = map_reduce_chain.run(split_docs)
                return [summary, cb]
    except InvalidRequestError as er:
        raise InvalidRequestError(f"model: {llm.model_name}\ntoken_count: {token_count}", er.param)
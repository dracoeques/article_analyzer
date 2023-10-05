from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import load_prompt, PipelinePromptTemplate, PromptTemplate
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

    llm = ChatOpenAI(temperature=0, openai_api_key=apikey, model = 'gpt-3.5-turbo', max_tokens=1248)

    if token_count < 1248:
        # chunk size of template was calculated (3072-1600)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1248, chunk_overlap=0, model_name='gpt-3.5-turbo'
        )
        # Create Document object for the text
        docs = [Document(page_content=content)]
        split_docs = text_splitter.split_documents(docs)
    else:
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=11088, chunk_overlap=0, model_name='gpt-3.5-turbo-16k',
        )
        if token_count >= 11088:
            text_splitter = TokenTextSplitter.from_tiktoken_encoder(
                chunk_size=11088, chunk_overlap=0, model_name='gpt-3.5-turbo-16k',
            )
        # Create Document object for the text
        docs = [Document(page_content=content)]
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
        token_max=13333,
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

def categorize(apikey: str, primaries: list[str], secondaries: list[str]):
    primary = ""
    for i, title in enumerate(primaries):
        primary += f"{i+1} {title}\n"
    
    secondary = ""
    for title in secondaries:
        secondary += f"- {title}\n"
    
    prompt = load_prompt("./prompts/categorize.yaml")
    # print(prompt.format(primary_titles=primary, secondary_titles=secondary))
    llm = ChatOpenAI(temperature=0, openai_api_key=apikey, model = 'gpt-3.5-turbo-16k')
    chain = LLMChain(llm=llm, prompt=prompt)
    example = """[{"Primary": "Trump Indicted for Espionage", "Secondary": ["Trump Indicted for Espionage", "Trump faces criminal charges", "Trump Arrested on Classified Documents Charges"], "Title": [Trump under investigation]}, ...]"""
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    token_count = len(encoding.encode(primary+secondary))
    print(primary)
    print(secondary)
    print(token_count)
    with get_openai_callback() as cb:
        result: str = chain.run(primary_titles=primary, secondary_titles=secondary, example=example)
        result = result.split(']}]')[0] + ']}]'
        return [result, cb]

def extra_research(apikey: str, articles: list[str]):
    content = "\n".join(articles)
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(
        chunk_size=10000, chunk_overlap=0, model_name='gpt-3.5-turbo-16k',
    )
    # Create Document object for the text
    docs = [Document(page_content=content)]
    split_docs = text_splitter.split_documents(docs)
    llm = ChatOpenAI(temperature=0, openai_api_key=apikey, model = 'gpt-3.5-turbo-16k', max_tokens=3696)

    original_prompt = load_prompt("./prompts/extra-research.yaml")
    full_template="""{{rules}}
    Output must be in this format. This must be python dictinoary or json object
    ###Output Format###
    {"Introduction": "An opening section that provides an overview of the topic and sets the context for the rest of the content.", "Historical Perspective": "A section that explores the past events, developments, or context relevant to the subject matter.", "People/entities Involved": "An exploration of the individuals or organizations that play a significant role in the topic.", "Motivations": "An analysis of the reasons, factors, or driving forces behind certain actions or decisions related to the topic.", "Recent Developments": "An examination of the most recent updates, advancements, or changes related to the subject.", "Impact": "An assessment of the consequences, effects, or influence the topic has on individuals, society, or other aspects.", "Examples from recent history": "Specific instances or cases from recent times that illustrate and support the topic under discussion.", "Conclusion": "A closing section that summarizes the main points and findings and may offer insights or suggestions for the future."}

    """
    final_prompt = PromptTemplate.from_template(full_template, template_format="jinja2")
    input_prompts = [
        ("rules", original_prompt),
    ]
    prompt = PipelinePromptTemplate(final_prompt=final_prompt, pipeline_prompts=input_prompts)

    # Map
    map_chain = LLMChain(llm=llm, prompt=prompt)

    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="articles", 
        # reduce_k_below_max_tokens=True,
    )
    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=13333,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="articles",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    with get_openai_callback() as cb:
        if len(split_docs) == 1:
            summary = combine_documents_chain.run(split_docs)
            return [summary, cb]
        else:
            summary = map_reduce_chain.run(split_docs)
            return [summary, cb]

def deep_research(apikey: str, articles: list[dict[str:str]], background: dict):
    content = "\n".join([f"Title: {article['title']}\nContent: {article['content']}" for article in articles])
    background = "\n".join([f"{p}: {v}\n" for p, v in background.items()])


    original_prompt = load_prompt("./prompts/deep-research.yaml")
    full_template="""{{rules}}
    Output must follow this format
    Output must be in this format. This must be python dictinoary or json object
    Don't include " in the middle of result sentences
    ###Output Format###
    {"1 day timeframe": {"Most likely": {"Description": "Most likely", "Explanation": "Explanation"}, "Possible": {"Description": "Possible", "Explanation": "Explanation"}, "Unlikely": {"Description": "Unlikely", "Explanation": "Explanation"}}, "1 week timeframe": {"Most likely": {"Description": "Most likely", "Explanation": "Explanation"}, "Possible": {"Description": "Possible", "Explanation": "Explanation"}, "Unlikely": {"Description": "Unlikely", "Explanation": "Explanation"}}, "1 month timeframe": {"Most likely": {"Description": "Most likely", "Explanation": "Explanation"}, "Possible": {"Description": "Possible", "Explanation": "Explanation"}, "Unlikely": {"Description": "Unlikely", "Explanation": "Explanation"}}}

    ###additional background, context, and examples###
    {{background}}
    """
    final_prompt = PromptTemplate.from_template(full_template, template_format="jinja2")
    background_prompt = PromptTemplate.from_template(background, template_format="jinja2")
    input_prompts = [
        ("rules", original_prompt),
        ("background", background_prompt),
    ]
    prompt = PipelinePromptTemplate(final_prompt=final_prompt, pipeline_prompts=input_prompts)

    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    token_count = len(encoding.encode(prompt.format(articles='')))
    chunk_size = int((16000 - token_count) * 0.75)
    max_token = int((16000 - token_count) * 0.25)
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=0, model_name='gpt-3.5-turbo-16k',
    )
    # Create Document object for the text
    docs = [Document(page_content=content)]
    split_docs = text_splitter.split_documents(docs)
    llm = ChatOpenAI(temperature=0, openai_api_key=apikey, model = 'gpt-3.5-turbo-16k', max_tokens=max_token)

    # Map
    map_chain = LLMChain(llm=llm, prompt=prompt)

    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="articles", 
        # reduce_k_below_max_tokens=True,
    )
    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=13333,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="articles",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    with get_openai_callback() as cb:
        if len(split_docs) == 1:
            summary = combine_documents_chain.run(split_docs)
            return [summary, cb]
        else:
            summary = map_reduce_chain.run(split_docs)
            return [summary, cb]

def impactul_news(apikey: str, articles: list[dict]):
    content = "\n".join([f"Title: {article['title']}\nSummary: {article['summary']}" for article in articles])

    original_prompt = load_prompt("./prompts/impactful-news.yaml")
    full_template="""{{rules}}
    Output must follow this format
    Output must be in this format. This must be python dictinoary or json object
    Don't include " in the middle of result sentences
    ###Output Format###
    [{"title": title, "explanation": Explanation}, {"title": title, "explanation": Explanation}, ... {"title": title, "explanation": Explanation}]
    """
    final_prompt = PromptTemplate.from_template(full_template, template_format="jinja2")
    input_prompts = [
        ("rules", original_prompt),
    ]
    prompt = PipelinePromptTemplate(final_prompt=final_prompt, pipeline_prompts=input_prompts)

    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    token_count = len(encoding.encode(prompt.format(articles='')))
    chunk_size = int((16000 - token_count) * 0.75)
    max_token = int((16000 - token_count) * 0.25)
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=0, model_name='gpt-3.5-turbo-16k',
    )
    # Create Document object for the text
    docs = [Document(page_content=content)]
    split_docs = text_splitter.split_documents(docs)
    llm = ChatOpenAI(temperature=0, openai_api_key=apikey, model = 'gpt-3.5-turbo-16k', max_tokens=max_token)

    # Map
    map_chain = LLMChain(llm=llm, prompt=prompt)

    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="articles", 
        # reduce_k_below_max_tokens=True,
    )
    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=13333,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="articles",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    with get_openai_callback() as cb:
        if len(split_docs) == 1:
            summary = combine_documents_chain.run(split_docs)
            return [summary, cb]
        else:
            summary = map_reduce_chain.run(split_docs)
            return [summary, cb]

def prediction(apikey: str, topics: list[dict], at_glance: bool, timeframe: str):
    content = "\n".join([f"topic: {topic['topic']}\nprediction: {topic['prediction']}" for topic in topics])

    original_prompt = load_prompt("./prompts/prediction.yaml")
    time = '24 hours'
    condition = "on a 1-day time frame"    
    if timeframe == 'week':
        time = 'one week'
    if timeframe == 'month':
        time = 'one month'

    if at_glance:
        original_prompt = load_prompt("./prompts/prediction_at_glance.yaml")
        condition = "in general"    

    main_template = f"""Ya the prompt for stage 6 should be this one:
    Imagine you are a professional news analyst and journalist

    ###Task###
    Let's think step by step.

    Describe what you believe to be the next biggest development or emerging trend in the science and technology category of the news based on the predictions and summaries to the most relevant news topics {condition}.

    Your analysis should be based on current trends, ongoing research, industry news, or any other relevant information sources.

    ###Goal###
    Think of it as what to look out for in the coming days. A convergence of what will happen from the summaries, all of the topics, and most likely predictions in the next {time}. use information from all of the topics to justify your reasoning"""
    main_prompt = PromptTemplate.from_template(main_template, template_format="jinja2")
    full_template="""
    {{main}}
    {{rules}}
    #######
    Output must follow this format
    Output must be in this format. This must be python dictinoary or json object
    Don't include " in the middle of result sentences
    ###Output Format###
    {"Developing Trend 1": Developing_Trend_1, "Explanation": Explanation, "Opportunities that may arise": Opportunities_that_may_arise, "Potential Pitfalls": Potential_Pitfalls}}
    """
    final_prompt = PromptTemplate.from_template(full_template, template_format="jinja2")
    input_prompts = [
        ("main", main_prompt),
        ("rules", original_prompt),
    ]
    prompt = PipelinePromptTemplate(final_prompt=final_prompt, pipeline_prompts=input_prompts)

    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    token_count = len(encoding.encode(prompt.format(topics='')))
    chunk_size = int((16000 - token_count) * 0.75)
    max_token = int((16000 - token_count) * 0.25)
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=0, model_name='gpt-3.5-turbo-16k',
    )
    # Create Document object for the text
    docs = [Document(page_content=content)]
    split_docs = text_splitter.split_documents(docs)
    llm = ChatOpenAI(temperature=0, openai_api_key=apikey, model = 'gpt-3.5-turbo-16k', max_tokens=max_token)

    # Map
    map_chain = LLMChain(llm=llm, prompt=prompt)

    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="topics", 
        # reduce_k_below_max_tokens=True,
    )
    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=13333,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="topics",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    with get_openai_callback() as cb:
        if len(split_docs) == 1:
            summary = combine_documents_chain.run(split_docs)
            return [summary, cb]
        else:
            summary = map_reduce_chain.run(split_docs)
            return [summary, cb]
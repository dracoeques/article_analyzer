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

    original_prompt = load_prompt("./prompts/extra-research.yaml")
    full_template="""{{rules}}
    Please adhere to the following structure for your examination
    Output must be in this format. This must be valid python dictinoary or json object
    Don't include " in the middle of result sentences in the output
    Don't include ' in the middle of result sentences in the output
    Don't include line breaking character (new line character) in the middle of result sentences in the output
    ###Output Format###
    {"Introduction": "Summarize the articles' topic and content briefly. Highlight the key issues or events to be analyzed.", "Historical Context": "Discuss the events' history. Identify crucial factors, background details, and significant precedents relevant to the situation.", "Key Players": "Identify the primary individuals or organizations involved in the events. Explore their roles and their effects on the events.", "Underlying Motivations": "Examine the driving forces behind the actions in the articles. Dig into the goals, interests, or ideologies of the engaged parties, supporting your analysis with solid evidence or credible theories.", "Recent Developments": "Address any fresh updates or happenings related to the events.", "Impact": "Evaluate the immediate and long-term consequences of the events.", "Future Challenges": "Identify potential challenges that may arise from these events. Discuss potential implications and strategies to tackle them.", "Historical Comparisons": "Provide three instances from recent history that resonate with the events in the articles. Analyze each example, emphasizing similarities, differences, and lessons gleaned.", "Conclusion": "Wrap up your analysis by encapsulating the key findings or insights. Propose areas for further research, if applicable."}

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

def deep_research(apikey: str, articles: list[dict[str:str]], background: dict):
    content = "\n".join([f"Title: {article['title']}\nContent: {article['content']}" for article in articles])
    background = "\n".join([f"{p}: {v}\n" for p, v in background.items()])


    original_prompt = load_prompt("./prompts/deep-research.yaml")
    full_template="""{{rules}}
    Output must follow this format
    Output must be in this format.
    Don't include " in the middle of result sentences in output
    Output format must follow this format
    Output must be json object
    
    This must be valid python dictinoary or json object
    
    ###Output Format to follow###
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

def prediction(apikey: str, topics: list[dict], category: str, timeframe: str):
    content = "\n".join([f"topic: {topic['topic']}\nprediction: {topic['prediction']}" for topic in topics])

    main_prompt = load_prompt("./prompts/prediction.yaml")
    category_prompt = PromptTemplate.from_template(category, template_format="jinja2")
    timeframe_prompt = PromptTemplate.from_template(timeframe, template_format="jinja2")
    time = "on a 1-day time frame"    
    if timeframe == 'week':
        time = 'on a 1-week time frame'
    if timeframe == 'month':
        time = 'on a 1-month time frame'

    full_template="""
    Imagine you are a professional news analyst and journalist

    ###Task###
    Let's think step by step.

    Describe what you believe to be the next biggest development or emerging trend in the {{category}} category of the news based on the predictions and summaries to the most relevant news topics {{timeframe}}.
    {{main}}
    #######
    Output must follow this format
    Output must be in this format. This must be valid python dictinoary or json object
    Don't include " in the middle of result sentences
    The output for Explanation should explain how the developing trend came to be as well as describe in detail the potential connections, ripple effects, etc of the developing trend

    ###Output Format###
    {"Developing Trend 1": Developing_Trend_1, "Explanation": Explanation, "Opportunities that may arise": Opportunities_that_may_arise, "Potential Pitfalls": Potential_Pitfalls}}
    """
    final_prompt = PromptTemplate.from_template(full_template, template_format="jinja2")
    input_prompts = [
        ("main", main_prompt),
        ("category", category_prompt),
        ("timeframe", timeframe_prompt),
    ]
    prompt = PipelinePromptTemplate(final_prompt=final_prompt, pipeline_prompts=input_prompts)

    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    token_count = len(encoding.encode(prompt.format(topics='', category=category, timeframe=time)))
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
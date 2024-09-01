import sys
import os
sys.path.append('/content/tagging_and_extraction/')

import config
import wikipedia

from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional,List
import nest_asyncio

from langchain_google_vertexai import ChatVertexAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_huggingface import HuggingFaceEmbeddings



from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel
from langchain.schema.runnable import RunnableLambda
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv('/content/tagging_and_extraction/keys.env')

class main():

  def __init__(self):

    self.init_LLM_ = False
    self.bind_model_to_entities_ = False
    self.init_entity_extraction_chain_ = False

  def wikipedia_search(self,
                      query:str):
    try:
      metadata_dict = {}
      wiki_search = wikipedia.search(query,results = 1)
      for search_item in wiki_search:
        page = wikipedia.page(search_item, auto_suggest=False)
        temp_dict = {}
        temp_dict['summary'] = page.summary
        temp_dict['title'] = page.title
        temp_dict['url'] = page.url
        metadata_dict[search_item] = temp_dict
      return metadata_dict
    except wikipedia.exceptions.PageError as pe:
      return None
    except wikipedia.exceptions.DisambiguationError as de:
      return {'Select a specfic keywork from options': de.options}

  def load_pagecontent(self,
                      metadata_dict: dict):
    # print(metadata_dict)
    # Loading the HTML content
    nest_asyncio.apply() # This is to allow AsyncChromiumLoader to run event loops within jupitor's running event loop
    loader = AsyncChromiumLoader([ ele['url'] for ele in metadata_dict.values()])
    html = loader.load()
    # Transforming the HTML content
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p"])
    for idx,(key, value)  in enumerate(metadata_dict.items()):
      value['page_content'] = docs_transformed[idx].page_content

    return metadata_dict

  def init_LLM(self,
                   model_name: str = None):
    if model_name is None:
      model_name = "gemini-1.5-flash-001" #"gemini-1.5-pro-latest"
    self.llm = ChatGoogleGenerativeAI(model=model_name,
                                    google_api_key=os.getenv('GOOGLE_API_KEY_2'),
                                    max_output_tokens = 1024,
                                    temperature = 0.2,
                                    verbose=False,
                                    )
    self.init_LLM_ = True

  def bind_model_to_entities(self):

    if self.init_LLM_ is False:
      self.init_LLM()
      
    class SearchPerson(BaseModel):
      """ Extracts any person related information from a section of text."""
      name: str = Field(description="The name of the person")
      date_of_birth: Optional[str] = Field(description = "The date of birth of the person")
      place_of_birth: Optional[str] = Field(description = "The place of birth of the person")
      person_summary: str = Field(description="A brief summary about the person including their notable achievement")

      date_of_death: Optional[str] = Field(description = "The date of death of the person")
      place_of_death: Optional[str] = Field(description = "The place of death of the person")

    class SearchEntity(BaseModel):
      """ Extracts any entity related information from a section of text."""
      name: str = Field(description = " The name of the entity")
      type_of_entity: str = Field(description = "WHat type of entity is it. Exaple: Organization, Institution, Location")
      entity_summary: str = Field(description = "A brief summary about the entity")


    class SearchEvents(BaseModel):
      """ Extracts any event related information from a section of text."""
      name: str = Field(description = "The name of the event")
      date: str = Field(description = "The date of the event")
      location: str = Field(description = "The location of the event")
      event_summary: str = Field(description = "A brief summary about the event")

    self.llm_with_functions = self.llm.bind(
    functions = [convert_pydantic_to_openai_function(func) for func in [SearchPerson,SearchEntity,SearchEvents]]
                  )
    self.bind_model_to_entities_ = True

  def init_entity_extraction_chain(self,
                                   run_extraction_on: str = 'summary'):

    assert run_extraction_on in ['summary','page_content'], "run_extraction_on must be either 'summary' or 'page_content'"
    if self.bind_model_to_entities_ is False:
      self.bind_model_to_entities()
    person_template = """
      Extract all the person related information from the following context. Do not make up any information.Partial information is acceptable.
      {input}

    """

    entity_template = """
      Extract all the entity related information from the following context. Do not make up any information.Partial information is acceptable.
      {input}

    """

    event_template = """
      Extract all the event related information from the following context. Do not make up any information.Partial information is acceptable.
      {input}

    """

    person_prompt = ChatPromptTemplate.from_template(person_template)
    entity_prompt = ChatPromptTemplate.from_template(entity_template)
    event_prompt = ChatPromptTemplate.from_template(event_template)


    person_chain = person_prompt | self.llm_with_functions
    entity_chain = entity_prompt | self.llm_with_functions
    event_chain = event_prompt | self.llm_with_functions 

    parser_chain = RunnablePassthrough() |  JsonOutputFunctionsParser().map()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024,chunk_overlap=0)
    split_text = RunnableLambda(lambda x:[{'input': doc} for doc in text_splitter.split_text('\n\n'.join([ele[run_extraction_on] for ele in x.values()]))])
    

    self.master_chain = RunnableLambda(self.wikipedia_search) | self.load_pagecontent | split_text | \
    RunnableParallel( {'person':person_chain.map(),
                      'entity': entity_chain.map(),
                      'event': event_chain.map()} ) |\
    RunnableLambda(lambda x:{key: parser_chain.invoke(val) for key,val in x.items()})
    self.init_entity_extraction_chain_ = True

  def extract_entity(self,
                     query:str,):
    if self.init_entity_extraction_chain_ is False:
      self.init_entity_extraction_chain()
    return self.master_chain.invoke(query)
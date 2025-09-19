import langgraph
import boto3
import json
import pandas as pd
import os
import ast
from typing import TypedDict, Dict, List
import requests
from langgraph.graph import StateGraph, END, START
import logging
import shelve
import time
import random
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate

logging.getLogger("langchain_aws.chat_models.bedrock_converse").setLevel(logging.WARNING)
logging.getLogger("langchain_aws").setLevel(logging.WARNING)

class InformationExtractionAgent:
    def __init__(self, secrets_path='secrets.json', region='us-west-2', cache_path='wiki_cache.db'):
        # Load secrets
        with open(secrets_path, 'r') as file:
            secrets = json.load(file)
            self.aws_access_key_id = secrets.get('AWS_ACCESS_KEY')
            self.aws_secret_access_key = secrets.get('AWS_SECRET_ACCESS_KEY')
            self.wiki_key = secrets.get('WIKIPEDIA_KEY')

        # AWS session and Bedrock client
        session = boto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=region
        )

        self.model_id = 'anthropic.claude-3-5-sonnet-20241022-v2:0' #'anthropic.claude-3-opus-20240229-v1:0'
        self.llm = ChatBedrockConverse(
                model=self.model_id,
                client=session.client("bedrock-runtime")
            )

        self.max_rpm = 100 
        self.seconds_per_request = 50 / self.max_rpm
        self._last_call_time = 0  # tracks last invocation time

        # Logging setup
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)
        self.logger.disabled = True

        # Build graph
        self.compiled_graph = self._build_graph()
        self._save_graph_image("information_extraction_graph.png")

        # Cache setup
        self.cache_path = cache_path


    def call_claude(self, messages: List[dict]) -> str:

        # Pacing: wait if last call was too recent
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self.seconds_per_request:
            time.sleep(self.seconds_per_request - elapsed)

        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                response = self.llm.invoke(messages)
                self._last_call_time = time.time()  # update after success
                return response.content
            except ClientError as e:
                if e.response['Error']['Code'] in ("ThrottlingException", "TooManyRequestsException"):
                    wait_time = random.uniform(5, 15)
                    print(f"Throttled. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    raise
        raise Exception("Max retries exceeded for Bedrock API calls.")

    def get_wikipedia_intro(self, title: str) -> Dict[str, List]:
        """
        Fetch the introductory paragraph(s) from a Wikipedia page for a given title.
        Save results in a local cache to avoid redundant API calls.
        
        Args:
            title (str): The Wikipedia page title to query.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'introduction': str of extracted intro text lines.
                - 'information': List[str] of extracted intro text lines.
                - 'metadata': str indicating status ('one entry', 'multiple entries',
                            'no info', or 'respond error').
        """

        # Check cache first
        with shelve.open(self.cache_path) as cache:
            if title in cache:
                return cache[title]
                
            headers = {'Authorization': self.wiki_key, 'User-Agent': 'thesis'}
            endpoint = 'https://en.wikipedia.org/w/api.php'
            params = {
                'format': 'json',
                'action': 'query',
                'prop': 'extracts|pageprops',
                'exintro': 1,
                'explaintext': 1,
                'redirects': 1,
                'titles': title,
            }
    
            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            
            if not response.ok:
                return {'information': [], 'metadata': 'respond error'}
            
            data = response.json()
            page = next(iter(data['query']['pages'].values()))
            extract = page.get('extract', '')
            pageprops = page.get('pageprops', [])
            
            is_disambiguation = any('disambiguation' in str(cat).lower() for cat in pageprops)
            if not extract:
                return {'information': [], 'metadata': 'no info'}
            
            if is_disambiguation:
                metadata = 'multiple entries'
                intro_list = extract
            else:
                intro_list = self.get_wikipedia_summary(title)
                metadata = 'one entry'
            
            result = {'introduction': intro_list, 'information': extract, 'metadata': metadata}
            cache[title] = result # Save to cache
        return result

    def get_wikipedia_summary(self, title: str) -> str:
        """
        Fetch a short summary for a Wikipedia page.

        Args:
            title (str): The Wikipedia page title.

        Returns:
            str: The short summary text, or an empty string if not found.
        """
        title = title.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        response = requests.get(url, headers={"User-Agent": "thesis"}, timeout=10)
        if not response.ok:
            return ""
        data = response.json()
        return data.get("extract", "")

    class AgentState(TypedDict):
        text: str
        name: str
        contains_name: bool
        wiki_search: Dict[str, List]
        name_mapping: bool
        tailored_summary: str
        final_state: str

    def detect_name_node(self, state: 'InformationExtractionAgent.AgentState') -> 'InformationExtractionAgent.AgentState':
        sentence = state["text"]
        entity = state["name"]
        prompt = ChatPromptTemplate.from_messages([
            ("user", "Does entity '{entity}' in the following text refer to a person's name? Respond only with true or false.\n'{sentence}'")
        ])
        messages = prompt.format_messages(entity=entity, sentence=sentence)
        result = self.call_claude(messages).strip().lower()
        state["contains_name"] = result == "true"
        if state["contains_name"]:
            self.logger.info("Name detected")
        else:
            self.logger.info("No name detected")
            state['final_state'] = 'No name detected'
        return state

    def get_wikipedia_intro_node(self, state: 'InformationExtractionAgent.AgentState') -> 'InformationExtractionAgent.AgentState':
        state["wiki_search"] = self.get_wikipedia_intro(state["name"])
        if state["wiki_search"]["metadata"] != "no info":
            self.logger.info("Information retrieved")
        else:
            self.logger.info("Not such entity on Wikipedia")
            state['final_state'] = 'No wiki entry'
        return state

    def route_after_detect(self, state: 'InformationExtractionAgent.AgentState') -> str:
        return "has_name" if state["contains_name"] else "no_name"

    def check_info_relevancy_node(self, state: 'InformationExtractionAgent.AgentState') -> 'InformationExtractionAgent.AgentState':
        sentence = state["text"]
        wiki_info = " ".join(state["wiki_search"]['introduction'])
        prompt = ChatPromptTemplate.from_messages([
                ("user", 
                "Is the following Wikipedia info relevant to identifying the entity '{entity}' in the context of this text? "
                "Respond only with true or false. Say false if the wiki info doesn't help to determine the identity of the entity.\n"
                "TEXT: {sentence}\nWIKI INFO: {wiki_info}")
            ])
        messages = prompt.format_messages(entity=state['name'], sentence=sentence, wiki_info=wiki_info)
        result = self.call_claude(messages).strip().lower()
        state["name_mapping"] = result == "true"
        if state["name_mapping"]:
            self.logger.info("Name matched with retrieved info")
        else:
            self.logger.info("Name didn't match with retrieved info")
            state['final_state'] = 'Not matched info'
        return state
    
    def route_after_mapping(self, state: 'InformationExtractionAgent.AgentState') -> str:
        return "mapping_true" if state["name_mapping"] else "mapping_false"

    def route_after_wiki_search(self, state: 'InformationExtractionAgent.AgentState') -> str:
        return "info_not_found" if state["wiki_search"]['metadata'] == 'no info' else "info_found"

    def tailor_summary_node(self, state: 'InformationExtractionAgent.AgentState') -> 'InformationExtractionAgent.AgentState':
        sentence = state["text"]
        wiki_info = " ".join(state["wiki_search"]['information'])
        prompt = ChatPromptTemplate.from_messages([
            ("user",
            "You are given:\n"
            "1. A sentence from a text.\n"
            "2. Wikipedia information about the entity '{entity}'.\n"
            "Your task:\n"
            "- Assume the reader (another LLM) does not know who this entity is.\n"
            "- From the Wikipedia info, select only the facts needed to:\n"
            "a) Identify who the entity is.\n"
            "b) Mention relevant information that could be used as a context for the sentence.\n"
            "- Ignore unrelated details that do not help interpret the sentence.\n"
            "- Combine these into a single, concise, self-contained summary that would let the reader fully understand the reference in the sentence.\n\n"
            "TEXT: {sentence}\nWIKIPEDIA INFO: {wiki_info}\n\nReturn only the summary, no extra commentary.")
        ])
        messages = prompt.format_messages(entity=state['name'], sentence=sentence, wiki_info=wiki_info)

        result = self.call_claude(messages).strip()
        state["tailored_summary"] = result
        self.logger.info("Tailored information to text.")
        state['final_state'] = 'Tailored info'
        return state

    def _build_graph(self):
        myGraph = StateGraph(self.AgentState)
        
        myGraph.add_node("detect_name_node", self.detect_name_node)
        myGraph.add_node("get_wikipedia_intro_node", self.get_wikipedia_intro_node)
        myGraph.add_node("check_info_relevancy_node", self.check_info_relevancy_node)
        myGraph.add_node("tailor_summary_node", self.tailor_summary_node)
        
        myGraph.add_edge(START, "detect_name_node")   
        myGraph.add_conditional_edges(
            "detect_name_node",
            self.route_after_detect,
            {
                "has_name": "get_wikipedia_intro_node",
                "no_name": END
            }
        )
        myGraph.add_conditional_edges(
            "get_wikipedia_intro_node",
            self.route_after_wiki_search,
            {
                "info_found": "check_info_relevancy_node",
                "info_not_found": END
            }
        )
        myGraph.add_conditional_edges(
            "check_info_relevancy_node",
            self.route_after_mapping,
            {
                "mapping_true": "tailor_summary_node",
                "mapping_false": END
            }
        )
        myGraph.add_edge("tailor_summary_node", END)
        
        return myGraph.compile()

    def run(self, name: str, text: str):
        result = self.compiled_graph.invoke({"name": name, "text": text})
        return result
    
    def _save_graph_image(self, filename: str):
        image_data = self.compiled_graph.get_graph().draw_mermaid_png()
        with open(filename, mode="wb") as f:
            f.write(image_data)
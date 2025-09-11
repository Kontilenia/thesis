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

class InformationExtractionAgent:
    def __init__(self, secrets_path='secrets.json', region='us-west-2'):
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

        # Create a Bedrock client
        self.bedrock_client = session.client('bedrock-runtime')
        self.model_id = 'anthropic.claude-3-opus-20240229-v1:0'

        # Logging setup
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)  # Only warnings and errors will be logged
        file_handler = logging.FileHandler("agent.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

        # Build graph
        self.compiled_graph = self._build_graph()
        self._save_graph_image("information_extraction_graph.png")


    def call_claude(self, messages: list) -> str:
        body = {
            "messages": messages,
            "max_tokens": 300,
            "anthropic_version": "bedrock-2023-05-31"
        }
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']

    def get_wikipedia_intro(self, title: str) -> Dict[str, List]:
        """
        Fetch the introductory paragraph(s) from a Wikipedia page for a given title.

        Args:
            title (str): The Wikipedia page title to query.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'introduction': str of extracted intro text lines.
                - 'information': List[str] of extracted intro text lines.
                - 'metadata': str indicating status ('one entry', 'multiple entries',
                            'no info', or 'respond error').
        """

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
        
        return {'introduction': intro_list, 'information': extract, 'metadata': metadata}

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

    def detect_name_node(self, state: 'InformationExtractionAgent.AgentState') -> 'InformationExtractionAgent.AgentState':
        sentence = state["text"]
        entity = state["name"]
        messages = [        
            {"role": "user",
            "content": f"""Does entity '{entity}' in the following text 
            refer to a person's name? Respond only with true or false.
            '{sentence}'"""}
        ]
        result = self.call_claude(messages).strip().lower()
        state["contains_name"] = result == "true"
        if state["contains_name"]:
            self.logger.info("Name detected")
        else:
            self.logger.info("No name detected")
        return state

    def get_wikipedia_intro_node(self, state: 'InformationExtractionAgent.AgentState') -> 'InformationExtractionAgent.AgentState':
        state["wiki_search"] = self.get_wikipedia_intro(state["name"])
        self.logger.info("Information retrieved")
        return state

    def route_after_detect(self, state: 'InformationExtractionAgent.AgentState') -> str:
        return "has_name" if state["contains_name"] else "no_name"

    def check_info_relevancy_node(self, state: 'InformationExtractionAgent.AgentState') -> 'InformationExtractionAgent.AgentState':
        sentence = state["text"]
        wiki_info = " ".join(state["wiki_search"]['introduction'])
        messages = [{
            "role": "user",
            "content": f"""Is the following Wikipedia info relevant to the entity '{state['name']}' in the context of this text? Respond only with true or false.
            TEXT: {sentence}
            WIKI INFO: {wiki_info}"""
        }]
        result = self.call_claude(messages).strip().lower()
        state["name_mapping"] = result == "true"
        if state["name_mapping"]:
            self.logger.info("Name matched with retrieved info")
        else:
            self.logger.info("Name didn't match with retrieved info")
        return state

    def tailor_summary_node(self, state: 'InformationExtractionAgent.AgentState') -> 'InformationExtractionAgent.AgentState':
        sentence = state["text"]
        wiki_info = " ".join(state["wiki_search"]['information'])
        messages = [{
            "role": "user",
            "content": f"""You are given:
            1. A sentence from a text.
            2. Wikipedia information about the entity '{state['name']}'.
            Your task:
            - Assume the reader (another LLM) does not know who this entity is.
            - From the Wikipedia info, select only the facts needed to:
            a) Identify who the entity is.
            b) Mention relevant information that could be used as a context for the sentence.
            - Ignore unrelated details that do not help interpret the sentence.
            - Combine these into a single, concise, self-contained summary that would let the reader fully understand the reference in the sentence.

            TEXT: {sentence}
            WIKIPEDIA INFO: {wiki_info}

            Return only the summary, no extra commentary.
            """
        }]
        result = self.call_claude(messages).strip()
        state["tailored_summary"] = result
        self.logger.info("Tailored information to text.")
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
        
        myGraph.add_edge("get_wikipedia_intro_node", "check_info_relevancy_node")
        myGraph.add_edge("check_info_relevancy_node", "tailor_summary_node")
        myGraph.add_edge("tailor_summary_node", END)
        
        return myGraph.compile()

    def run(self, name: str, text: str):
        self.logger.info(f"New run:")
        result = self.compiled_graph.invoke({"name": name, "text": text})
        return result
    
    def _save_graph_image(self, filename: str):
        image_data = self.compiled_graph.get_graph().draw_mermaid_png()
        with open(filename, mode="wb") as f:
            f.write(image_data)

# Example usage:
agent = InformationExtractionAgent()
result = agent.run(name="John", text="I wish I was an Asian guy")
print(result)
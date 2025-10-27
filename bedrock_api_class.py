import boto3, json, time, random
from botocore.exceptions import ClientError
import boto3
import json
import time
import random
from botocore.exceptions import ClientError

class BedrockAPICalls:
    def __init__(self, secrets_path='secrets.json', region='us-west-2',
                 model_id='anthropic.claude-3-sonnet-20240229-v1:0'):
        # Load secrets (optional if using IAM roles)
        with open(secrets_path, 'r') as file:
            secrets = json.load(file)
            self.aws_access_key_id = secrets.get('AWS_ACCESS_KEY')
            self.aws_secret_access_key = secrets.get('AWS_SECRET_ACCESS_KEY')

        session = boto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=region
        )

        self.model_id = model_id
        self.client = session.client("bedrock-runtime")

        # Rate limiting
        self.max_rpm = 60   # adjust to your quota
        self.seconds_per_request = 60 / self.max_rpm
        self._last_call_time = 0

    def call_llm(self, prompt: str) -> str:
        # Pacing
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self.seconds_per_request:
            time.sleep(self.seconds_per_request - elapsed)

        retry_count = 0
        while retry_count < 5:
            try:

                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps({
                        # Anthropic
                        # "anthropic_version": "bedrock-2023-05-31",
                        # "messages": [{"role": "user", 
                        #               "content": prompt}],
                        # "max_tokens": 10,
                        # "temperature": 0.1

                        # Llama 3.1
                        # "prompt": prompt,
                        # "max_gen_len": 5,
                        # "temperature": 0.1

                        # Mistral
                        "prompt": prompt,
                        "max_tokens": 10,
                        "temperature": 0.1
                    }),
                    contentType="application/json",
                    accept="application/json"
                )
                self._last_call_time = time.time()
                result = json.loads(response["body"].read())
                return result['choices'][0]['message']['content'] #['content'][0]['text']  # ['generation']
            except ClientError as e:
                if e.response['Error']['Code'] in ("ThrottlingException", "TooManyRequestsException"):
                    wait_time = random.uniform(5, 15)
                    print(f"Throttled. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    raise
        raise Exception("Max retries exceeded for Bedrock API calls.")
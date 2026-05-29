from langchain_aws import ChatBedrockConverse
import os

class AWSBedrockLLM:
    def __init__(self, user_controls_input):
        self.user_controls_input = user_controls_input

    def get_llm_model(self):
        try:
            # We fetch the model ID passed in, falling back to Amazon Nova Lite
            model_id = self.user_controls_input.get("selected_aws_model", "amazon.nova-lite-v1:0")
            region = os.environ.get("AWS_REGION", "us-east-1")
            
            # ChatBedrockConverse uses the new Converse API which is recommended for RAG and Tool calling
            llm = ChatBedrockConverse(
                model=model_id,
                region_name=region,
                temperature=0,
                max_tokens=1024,
            )
            return llm
        except Exception as e:
            raise ValueError(f"Error initializing AWS Bedrock LLM: {e}")

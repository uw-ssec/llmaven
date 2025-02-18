import asyncio
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

class LLMInference:
    def __init__(self, model, tokenizer, generation_config: dict = None):
        """
        Initialize the inference pipeline.
        
        Parameters:
            model: The model instance.
            tokenizer: The corresponding tokenizer.
            generation_config (dict, optional): Override generation parameters.
        """
        generation_config = generation_config or {}
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=generation_config.get("max_new_tokens", 512),
            temperature=generation_config.get("temperature", 0.8),
            do_sample=generation_config.get("do_sample", True),
            return_full_text=generation_config.get("return_full_text", False)
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    async def generate_response(self, prompt: str) -> str:
        """
        Asynchronously generate a response from the prompt.
        """
        loop = asyncio.get_event_loop()
        # Run the synchronous pipeline call in an executor.
        result = await loop.run_in_executor(None, lambda: self.llm(prompt))
        # Assumes result is a list of dicts with key 'generated_text'
        return result[0]["generated_text"]

# Convenience function 
async def generate_response_fn(model, tokenizer, prompt, generation_config: dict = None) -> str:
    inference = LLMInference(model, tokenizer, generation_config)
    return await inference.generate_response(prompt)
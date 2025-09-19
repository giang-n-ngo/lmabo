import random
import re
import time
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from google.generativeai.types import generation_types

from key import API_KEYS

def check_available_model():
    # List all available models
    print("Listing available models and their supported methods:")
    for m in genai.list_models():
        # Check if the model supports the 'generateContent' method
        if 'generateContent' in m.supported_generation_methods:
            print(f"  Model Name: {m.name}, Supported Methods: {m.supported_generation_methods}")
        else:
            print(f"  Model Name: {m.name}, (Does NOT support generateContent)")

def test_api_key(key):
    """
    Test if a Gemini API key is valid by attempting a simple chat interaction.
    
    Args:
        key (str): API key to test
        
    Returns:
        bool: True if key is valid, False if it raises any errors
    """
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        chat = model.start_chat()
        response = chat.send_message("Test message")
        return True
    except Exception as e:
        print(f"Key failed: {str(e)}")
        return False

def get_valid_key():
    """
    Test all API keys and return only the valid ones.
    
    Returns:
        list: List of valid API keys
    """
    shuffled_keys = API_KEYS.copy()  # Create a copy to avoid modifying original
    random.shuffle(shuffled_keys)
    valid_key = None
    for key in shuffled_keys:
        if test_api_key(key):
            valid_key = key
            break
    if valid_key is None:
        print("No valid API keys found. Please check your keys and network connection.")
        exit()
    else:
        print(f"Using valid key: {valid_key[:8]}...")
        return valid_key

def configure_and_start_chat_api(first_prompt):
    valid_key = get_valid_key()
    genai.configure(api_key=valid_key)
    # init LLM
    model = genai.GenerativeModel(
        'gemini-2.5-flash-preview-05-20', 
    )
    # --- START THE CHAT SESSION ---
    print("Starting Gemini chat session with initial context...")
    try:
        chat = model.start_chat(history=[
            {"role": "user", "parts": [first_prompt]}
        ])
        # The first response from the model just confirms it understands the context
        # You might want to print/log this response, or just ignore it
        initial_response = chat.send_message("Do you understand the context?")
        print(f"Gemini's initial acknowledgement: {initial_response.text.strip()}")
        return chat, initial_response.text.strip()
    except Exception as e:
        print(f"Error starting chat or initial acknowledgement: {e}")
        print("Please check your API key, model availability, and network connection.")
        exit() # Exit if we can't even start the chat    

class QwenChatbot:
    def __init__(self, model_name, hosted=False, server_node="localhost"):
        """
        Initialize the chatbot with vLLM.
        
        Args:
            model_name: Hugging Face model name/path
        """
        self.hosted = hosted
        self.max_tokens = 8192  # Increased from 4096 to leave more room for context
        self.top_p = 0.9  # Nucleus sampling probability
        self.model_name = model_name
        print(f"Loading model: {self.model_name}")
        if hosted:
            from openai import OpenAI
            openai_api_key = "EMPTY"
            openai_api_base = f"http://{server_node}:8000/v1"

            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            print(f"Using hosted vLLM API at {server_node}:8000")
        else:
            from vllm import LLM, SamplingParams
            # Initialize vLLM with optimized settings
            self.llm = LLM(
                model=self.model_name,
                gpu_memory_utilization=0.3,  # Use 80% of GPU memory
                max_model_len=16384,         # Maximum sequence length
                dtype="float16",             # Use half precision for efficiency
                trust_remote_code=True,      # Required for some models
                tensor_parallel_size=1,      # Number of GPUs (set to 1 for single GPU)
            )
            
            # Sampling parameters for generation
            self.sampling_params = SamplingParams(
                temperature=0.0,             # Controls randomness (0.0 = deterministic)
                top_p=self.top_p,                   # Nucleus sampling
                max_tokens=self.max_tokens,              # Maximum tokens to generate
                repetition_penalty=1.1,      # Reduce repetition
            )
            
            # Get tokenizer for special tokens
            self.tokenizer = self.llm.get_tokenizer()
            
            # Update sampling params with proper stop tokens
            self.sampling_params.stop_token_ids = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|im_end|>")  # Qwen's chat end token
            ]
        
        # Conversation history
        self.history = []
        
        print("Model loaded successfully!")

    def _format_chat_prompt(self, user_message):
        """
        Format the conversation history into a proper chat prompt for Qwen.
        
        Args:
            user_message: New user message to add
            
        Returns:
            Formatted prompt string
        """
        # Add the new user message to history
        self.history.append({"role": "user", "content": user_message})
        
        # Build the chat prompt using Qwen's format
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        
        for message in self.history:
            role = message["role"]
            content = message["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Add assistant start token
        prompt += "<|im_start|>assistant\n"
        
        return prompt

    def _clean_response(self, response_text):
        """
        Clean the generated response by removing unwanted tokens and formatting.
        
        Args:
            response_text: Raw response from the model
            
        Returns:
            Cleaned response text
        """
        # check if response_text is string
        if isinstance(response_text, str):
            # Remove everything between <think> and </think> tags (including newlines)
            cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            # Remove any remaining <think> or </think> tags
            cleaned = re.sub(r'</?think>', '', cleaned)
            # Clean up extra whitespace
            cleaned = re.sub(r'\n\s*\n', '\n', cleaned)  # Remove empty lines
            cleaned = cleaned.strip()
            return cleaned
        else:
            return ""
    
    def _manage_context_length(self, prompt):
        """
        Manage conversation history to prevent context overflow.
        If the prompt is too long, remove older messages from history.
        
        Args:
            prompt: The formatted prompt string
            
        Returns:
            Adjusted prompt that fits within context limits
        """
        # Rough estimation: 1 token ≈ 4 characters for most languages
        # Leave some buffer for safety
        max_context_chars = 30000 * 4  # ~30k tokens in characters
        max_completion_chars = self.max_tokens * 4  # Reserve space for completion
        available_chars = max_context_chars - max_completion_chars
        
        if len(prompt) <= available_chars:
            return prompt
        
        print(f"Warning: Prompt too long ({len(prompt)} chars), trimming conversation history...")
        
        # Keep system prompt and recent messages
        system_part = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        assistant_start = "<|im_start|>assistant\n"
        
        # Remove older messages from history until prompt fits
        while len(prompt) > available_chars and len(self.history) > 2:
            # Remove the second oldest message (keep the most recent user message)
            if len(self.history) > 2:
                self.history.pop(1)  # Remove second message (keep first user message)
            
            # Rebuild prompt
            prompt = system_part
            for message in self.history:
                role = message["role"]
                content = message["content"]
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            prompt += assistant_start
        
        print(f"Trimmed prompt to {len(prompt)} characters with {len(self.history)} messages in history")
        return prompt
    
    def generate_response(self, user_message):
        """
        Generate a response to the user message.
        
        Args:
            user_message: User's input message
            
        Returns:
            Assistant's response
        """
        try:
            # Format the prompt with conversation history
            prompt = self._format_chat_prompt(user_message)
            
            # Manage context length to prevent overflow
            prompt = self._manage_context_length(prompt)
            if self.hosted:
                # call the client API for hosted models
                outputs = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    top_p=self.top_p,
                    extra_body={
                        "top_k": 20,
                    },
                )
                response = outputs.choices[0].message.content
                if "</think>" in response:
                    response = response.split("</think>")[1].strip()
            else:
                # Generate response using vLLM
                outputs = self.llm.generate([prompt], self.sampling_params)
                response = outputs[0].outputs[0].text
            
            # Clean the response
            cleaned_response = self._clean_response(response)
            
            # Add assistant response to conversation history
            self.history.append({"role": "assistant", "content": cleaned_response})
            
            return cleaned_response
            
        except Exception as e:
            print(f"Full error details: {type(e).__name__}: {e}")
            if self.hosted:
                print("Hosted mode connection failed. Check if vLLM server is running on localhost:8000")
            else:
                print("Local vLLM generation failed. Check GPU availability and memory.")
            exit()
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.history = []
        print("Conversation history cleared.")

    def get_history(self):
        """Get the current conversation history."""
        return self.history.copy()

def configure_and_start_chat_ops(first_prompt, server_node="localhost", ops_model_name="Qwen/Qwen3-8B"):
    # Load Qwen3 model and tokenizer from Hugging Face Hub
    chatbot = QwenChatbot(model_name=ops_model_name, hosted=True, server_node=server_node)
    print(f"Initialized {ops_model_name}")
    # Start a conversation
    response = chatbot.generate_response(first_prompt)
    print("Assistant:", response.strip())
    return chatbot

class ConversationHolder:
    def __init__(
        self,
        llm="api",
        first_prompt="",
        full_choice_list=[],
        server_node="localhost",  # Default to localhost if not specified,
        default_choice="UCB",
        ops_model_name="Qwen/Qwen3-8B",
    ):
        self.llm = llm
        self.full_choice_list = full_choice_list
        self.messages = []
        self.token_count = 0  # Initialize token count
        self.money_cost = 0.0  # Initialize money cost
        if self.llm == "api":
            self.chat, initial_response = configure_and_start_chat_api(first_prompt)
            self.messages.append(initial_response)
            self.api_initial_delay_seconds = 1
            self.api_max_entries = 10
            self.api_max_delay_seconds = 120
        elif self.llm == "ops":
            self.chatbot = configure_and_start_chat_ops(first_prompt, server_node, ops_model_name)
            self.messages.append(self.chatbot.history[-1]["content"])
        self.default_choice = default_choice

    def _api_process_suggestion_response(self, response_text):
        """
        Process the response text from the LLM to extract the suggested acquisition function (AF)
        and its justification.
        
        Args:
            response_text (str): The raw response text from the LLM.
        
        Returns:
            tuple: Suggested AF and its justification.
        """
        if ":" in response_text:
            response, justification = response_text.split(":", maxsplit=1)
            response = response.strip()
            justification = justification.strip()
        else:
            response = response_text.strip()
        if response not in self.full_choice_list:
            response = self.default_choice
            justification = "Nothing"
        print(f"LLM suggested AF: {response} justified by: {justification}")
        self.messages.append(response_text.strip())
        return response

    def _api_suggest_acq_type(self, prompt):
        retries = 0
        current_delay = self.api_initial_delay_seconds
        
        llm_suggested_af = self.default_choice
        while retries < self.api_max_entries:
            try:
                # Send the updated summary to the active chat
                response = self.chat.send_message(
                    prompt,
                    generation_config=generation_types.GenerationConfig(
                        temperature=0.0,
                    )
                )

                if response.text:
                    llm_suggested_af = self._api_process_suggestion_response(response.text)
                    # Update token count and cost (replace with actual logic)
                    input_tokens = len(prompt.split())  # Rough estimate
                    output_tokens = len(response.text.split())  # Rough estimate
                    self.token_count += input_tokens + output_tokens
                    # Replace with actual pricing
                    self.money_cost += (input_tokens * 0.3/1e6) + (output_tokens * 2.5/1e6) 
                    break # Success, exit retry loop

                else:
                    print("LLM returned no text content in response.")
                    self.messages.append("LLM returned no text content in response.")
                    llm_suggested_af = self.default_choice # Or handle as an error
                    break

            except ResourceExhausted as e:
                error_message = str(e) # Get the full string representation of the error
                suggested_delay_seconds = current_delay # Default to current backoff delay

                # Use regex to find the retry_delay from the error string
                match = re.search(r"retry_delay \{[\s\n]+seconds: (\d+)[\s\n]+\}", error_message)
                if match:
                    try:
                        suggested_delay_seconds = int(match.group(1))
                        print(f"API suggested waiting {suggested_delay_seconds} seconds (parsed from error message).")
                    except ValueError:
                        print("Could not parse suggested retry delay from error message. Using exponential backoff.")
                else:
                    print("No specific retry_delay found in error message. Using exponential backoff.")

                print(f"Rate limit hit (Retry {retries+1}/{self.api_max_entries}).")
                
                # Use the parsed suggested delay, or our exponential backoff
                wait_time = suggested_delay_seconds + random.uniform(0, suggested_delay_seconds * 0.1) # Add jitter
                wait_time = min(wait_time, self.api_max_delay_seconds) # Cap the wait time

                print(f"Waiting for {wait_time:.2f} seconds...")
                time.sleep(wait_time)

                retries += 1
                current_delay = min(current_delay * 2, self.api_max_delay_seconds) # Double delay for next retry

            except Exception as e:
                print(f"An unexpected error occurred during API call: {e}")
                break # Exit retry loop for other errors
        else:
            print(f"Failed to get LLM response after {self.api_max_entries} retries.")
            return "Intentional Incorrect AF"
        return llm_suggested_af  # Return the chat object and the response text for logging

    def _ops_process_suggestion_response(self, response_text):
        """
        Process the response text from the LLM to extract the suggested choice
        and its justification.
        
        Args:
            response_text (str): The raw response text from the LLM.
        
        Returns:
            str: Suggested choice
        """        
        # Extract choice and justification
        choice = self.default_choice
        justification = "Nothing"

        choice, justification = response_text.split(":", maxsplit=1)

        # Validate choice is in the allowed list
        if choice not in self.full_choice_list:
            print(f"Invalid choice '{choice}', using default '{self.default_choice}'")
            choice = self.default_choice

        print(f"LLM suggested choice: {choice} justified by: {justification}")
        self.messages.append(response_text)
        return choice

    def _ops_suggest_acq_type(self, prompt):
        llm_suggested_af = self.default_choice
        try:
            response = self.chatbot.generate_response(prompt)
            if response:
                llm_suggested_af = self._ops_process_suggestion_response(response.strip())
            else:
                print("LLM returned no text content in response.")
                llm_suggested_af = self.default_choice # Or handle as an error
        except Exception as e:
            print(f"An error occurred during LLM call: {e}")
            llm_suggested_af = "Intentional Incorrect AF"
        return llm_suggested_af

    def suggest_acq_type(self, prompt):
        if self.llm == "api":
            print("Total tokens used so far: ", self.token_count)
            print(f"Estimated cost so far: ${self.money_cost:.6f}")
            return self._api_suggest_acq_type(prompt)
        elif self.llm == "ops":
            return self._ops_suggest_acq_type(prompt)
        
    # def _api_suggest_choice(self, prompt):
    #     retries = 0
    #     current_delay = self.api_initial_delay_seconds
        
    #     llm_suggested_choice = self.default_choice
    #     while retries < self.api_max_entries:
    #         try:
    #             # Send the updated summary to the active chat
    #             response = self.chat.send_message(prompt)

    #             if response.text:
    #                 llm_suggested_choice = self._api_process_suggestion_response(response.text)
    #                 break # Success, exit retry loop

    #             else:
    #                 print("LLM returned no text content in response.")
    #                 self.messages.append("LLM returned no text content in response.")
    #                 llm_suggested_choice = self.default_choice # Or handle as an error
    #                 break

    #         except ResourceExhausted as e:
    #             error_message = str(e) # Get the full string representation of the error
    #             suggested_delay_seconds = current_delay # Default to current backoff delay

    #             # Use regex to find the retry_delay from the error string
    #             match = re.search(r"retry_delay \{[\s\n]+seconds: (\d+)[\s\n]+\}", error_message)
    #             if match:
    #                 try:
    #                     suggested_delay_seconds = int(match.group(1))
    #                     print(f"API suggested waiting {suggested_delay_seconds} seconds (parsed from error message).")
    #                 except ValueError:
    #                     print("Could not parse suggested retry delay from error message. Using exponential backoff.")
    #             else:
    #                 print("No specific retry_delay found in error message. Using exponential backoff.")

    #             print(f"Rate limit hit (Retry {retries+1}/{self.api_max_entries}).")
                
    #             # Use the parsed suggested delay, or our exponential backoff
    #             wait_time = suggested_delay_seconds + random.uniform(0, suggested_delay_seconds * 0.1) # Add jitter
    #             wait_time = min(wait_time, self.api_max_delay_seconds) # Cap the wait time

    #             print(f"Waiting for {wait_time:.2f} seconds...")
    #             time.sleep(wait_time)

    #             retries += 1
    #             current_delay = min(current_delay * 2, self.api_max_delay_seconds) # Double delay for next retry

    #         except Exception as e:
    #             print(f"An unexpected error occurred during API call: {e}")
    #             break # Exit retry loop for other errors
    #     else:
    #         print(f"Failed to get LLM response after {self.api_max_entries} retries.")
    #         return "Intentional Incorrect Choice"
    #     return llm_suggested_choice  # Return the chat object and the response text for logging

    # def _ops_suggest_choice(self, prompt):
    #     llm_suggested_choice = self.default_choice
    #     try:
    #         response = self.chatbot.generate_response(prompt)
    #         if response:
    #             llm_suggested_choice = self._ops_process_suggestion_response(response.strip())
    #             self.messages.append(response.strip())
    #         else:
    #             print("LLM returned no text content in response.")
    #             llm_suggested_choice = self.default_choice # Or handle as an error
    #     except Exception as e:
    #         print(f"An error occurred during LLM call: {e}")
    #         llm_suggested_choice = "Intentional Incorrect Choice"
    #     return llm_suggested_choice

    # def suggest_choice(self, prompt):
    #     if self.llm == "api":
    #         return self._api_suggest_choice(prompt)
    #     elif self.llm == "ops":
    #         return self._ops_suggest_choice(prompt)

    def _api_last_guess(self, last_prompt):
        try:
            response = self.chat.send_message(last_prompt)
            if response.text:
                print(response.text)
                self.messages.append(response.text.strip()) 
                return response.text.strip()
            else:
                print("No text guesses")
                return "No guesses"
        except ResourceExhausted:
            print("No more resources - no guessing")
            return "No more resources - no guessing"
        
    def _ops_last_guess(self, last_prompt):
        try:
            response = self.chatbot.generate_response(last_prompt)
            if response:
                print(response.strip())
                self.messages.append(response.strip())
                return response.strip()
            else:
                print("No text guesses")
                return "No guesses"
        except Exception as e:
            print(f"An error occurred during LLM call: {e}")
            return "Error in guessing"
    
    def last_guess(self, last_prompt):
        if self.llm == "api":
            return self._api_last_guess(last_prompt)
        elif self.llm == "ops":
            return self._ops_last_guess(last_prompt)
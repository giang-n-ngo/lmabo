import random
import re
import time
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from vllm import LLM, SamplingParams

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
    def __init__(self, model_name):
        """
        Initialize the chatbot with vLLM.
        
        Args:
            model_name: Hugging Face model name/path
        """
        print(f"Loading model: {model_name}")
        
        # Initialize vLLM with optimized settings
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.3,  # Use 80% of GPU memory
            max_model_len=4096,          # Maximum sequence length
            dtype="float16",             # Use half precision for efficiency
            trust_remote_code=True,      # Required for some models
            tensor_parallel_size=1,      # Number of GPUs (set to 1 for single GPU)
        )
        
        # Sampling parameters for generation
        self.sampling_params = SamplingParams(
            temperature=0.0,             # Controls randomness (0.0 = deterministic)
            top_p=0.9,                   # Nucleus sampling
            max_tokens=2048,              # Maximum tokens to generate
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
        # Remove everything between <think> and </think> tags (including newlines)
        cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        # Remove any remaining <think> or </think> tags
        cleaned = re.sub(r'</?think>', '', cleaned)
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)  # Remove empty lines
        cleaned = cleaned.strip()
        return cleaned
    
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
            
            # Generate response using vLLM
            outputs = self.llm.generate([prompt], self.sampling_params)
            
            # Extract the generated text
            response = outputs[0].outputs[0].text
            
            # Clean the response
            cleaned_response = self._clean_response(response)
            
            # Add assistant response to conversation history
            self.history.append({"role": "assistant", "content": cleaned_response})
            
            return cleaned_response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return error_msg
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.history = []
        print("Conversation history cleared.")

    def get_history(self):
        """Get the current conversation history."""
        return self.history.copy()

def configure_and_start_chat_ops(first_prompt):
    # Load Qwen3 model and tokenizer from Hugging Face Hub
    chatbot = QwenChatbot(model_name="Qwen/Qwen3-8B")
    print("Initialized Qwen3")
    # Start a conversation
    response = chatbot.generate_response(first_prompt)
    print("Assistant:", response.strip())
    return chatbot

class ConversationHolder:
    def __init__(
        self,
        llm="api",
        first_prompt="",
        full_acq_type_list=[]
    ):
        self.llm = llm
        self.full_acq_type_list = full_acq_type_list
        self.messages = []
        if self.llm == "api":
            self.chat, initial_response = configure_and_start_chat_api(first_prompt)
            self.messages.append(initial_response)
            self.api_initial_delay_seconds = 1
            self.api_max_entries = 10
            self.api_max_delay_seconds = 120
        elif self.llm == "ops":
            self.chatbot = configure_and_start_chat_ops(first_prompt)
            self.messages.append(self.chatbot.history[-1]["content"])
        self.default_af = "UCB"

    def _api_process_suggestion_response(self, response_text):
        """
        Process the response text from the LLM to extract the suggested acquisition function (AF)
        and its justification.
        
        Args:
            response_text (str): The raw response text from the LLM.
        
        Returns:
            tuple: Suggested AF and its justification.
        """
        print(response_text)
        if ":" in response_text:
            af, justification = response_text.split(":")
            af = af.strip()
            justification = justification.strip()
        else:
            af = response_text.strip()
            if af not in self.full_acq_type_list:
                af = self.default_af
            justification = "Nothing"
        print(f"LLM suggested AF: {af} justified by: {justification}")
        self.messages.append(response_text.strip())
        return af

    def _api_suggest_acq_type(self, prompt):
        retries = 0
        current_delay = self.api_initial_delay_seconds
        
        llm_suggested_af = self.default_af
        while retries < self.api_max_entries:
            try:
                # Send the updated summary to the active chat
                response = self.chat.send_message(prompt)

                if response.text:
                    llm_suggested_af = self._api_process_suggestion_response(response.text)
                    break # Success, exit retry loop

                else:
                    print("LLM returned no text content in response.")
                    self.messages.append("LLM returned no text content in response.")
                    llm_suggested_af = self.default_af # Or handle as an error
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
        Process the response text from the LLM to extract the suggested acquisition function (AF)
        and its justification.
        
        Args:
            response_text (str): The raw response text from the LLM.
        
        Returns:
            str: Suggested AF
        """        
        # Extract AF and justification
        af = self.default_af
        justification = "Nothing"
        
        af, justification = response_text.split(":", maxsplit=1)
        
        # Validate AF is in the allowed list
        if af not in self.full_acq_type_list:
            print(f"Invalid AF '{af}', using default '{self.default_af}'")
            af = self.default_af
        
        print(f"LLM suggested AF: {af} justified by: {justification}")
        self.messages.append(response_text)
        return af

    def _ops_suggest_acq_type(self, prompt):
        llm_suggested_af = self.default_af
        try:
            response = self.chatbot.generate_response(prompt)
            if response:
                llm_suggested_af = self._ops_process_suggestion_response(response.strip())
                self.messages.append(response.strip())
            else:
                print("LLM returned no text content in response.")
                llm_suggested_af = self.default_af # Or handle as an error
        except Exception as e:
            print(f"An error occurred during LLM call: {e}")
            llm_suggested_af = "Intentional Incorrect AF"
        return llm_suggested_af

    def suggest_acq_type(self, prompt):
        if self.llm == "api":
            return self._api_suggest_acq_type(prompt)
        elif self.llm == "ops":
            return self._ops_suggest_acq_type(prompt)
        
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
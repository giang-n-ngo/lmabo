import random
import re
import time
import torch
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from transformers import AutoModelForCausalLM, AutoTokenizer

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

class ChatHistory:
    def __init__(self):
        self.turns = []

    def add_turn(self, user, assistant):
        self.turns.append({"user": user, "assistant": assistant})

    def format_prompt(self, new_user_input=None):
        prompt = ""
        for turn in self.turns:
            prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        if new_user_input is not None:
            prompt += f"User: {new_user_input}\nAssistant:"
        return prompt
    
def model_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=1.0)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)    
    return response

def configure_and_start_chat_ops(first_prompt):
    # Load Qwen3 model and tokenizer from Hugging Face Hub
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    print("Initialized Qwen3")
    # Start a conversation
    history = ChatHistory()
    prompt = history.format_prompt(first_prompt)
    response = model_answer(model, tokenizer, prompt)
    print("Assistant:", response.strip())
    history.add_turn(first_prompt, response.strip())
    return model, tokenizer, history

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
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model, self.tokenizer, self.history = configure_and_start_chat_ops(first_prompt)
            self.messages.append(self.history.turns[0]["assistant"])
        self.default_af = "UCB"

    def _process_suggestion_response(self, response_text):
        """
        Process the response text from the LLM to extract the suggested acquisition function (AF)
        and its justification.
        
        Args:
            response_text (str): The raw response text from the LLM.
        
        Returns:
            tuple: Suggested AF and its justification.
        """
        if ":" in response_text:
            print(response_text)
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
                    llm_suggested_af = self._process_suggestion_response(response.text)
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
    
    def _ops_suggest_acq_type(self, prompt):
        llm_suggested_af = self.default_af
        try:
            response = model_answer(self.model, self.tokenizer, prompt)
            if response:
                llm_suggested_af = self._process_suggestion_response(response.strip())
                self.history.add_turn(prompt, response.strip())
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
            response = model_answer(self.model, self.tokenizer, last_prompt)
            if response:
                print(response.strip())
                self.history.add_turn(last_prompt, response.strip())
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
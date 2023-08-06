from constrainedGen import ConstrainedGeneration
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
class Online:
    def __init__(self, json_schema, model_name):
        # json schema to constrain generation to 
        self.json_schema = json_schema
        # HF model to load
        self.model_name = model_name
        try: 
            # Load model and tokenizer
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
        except:
            print("Error loading model and tokenizer")
        # Take care of loading for model through HF and tokenizer
        # load json builder
        self.builder = ConstrainedGeneration(
            model=model,
            tokenizer=tokenizer,
            json_schema=json_schema,
            max_string_token_length=20)
        


    
    def generate(self,prompt):
        '''
        Given a prompt generate a new text wrt the json schema specified
        '''
        # Generate text

        output = self.builder(prompt)

        return output







    def feedback(self, feedback):
        '''
        If online learning is enabled, then get user feedabck and 
        update the model online/zeroth order optimization

        TODO for v.0.1
        LORA hotswapping in the background since calculating the loss
        will add inference latency
        '''
        



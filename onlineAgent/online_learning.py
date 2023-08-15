from constrainedGen import ConstrainedGeneration
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
#from onlineAgent import feedback

class Online:
    def __init__(self, json_schema, model_name):
        # json schema to constrain generation to 
        self.json_schema = json_schema
        # HF model to load
        self.model_name = model_name
 
            # Load model and tokenizer
        try:
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
        
        except:
            print("Model not found")
            return
        try:

            # Take care of loading for model through HF and tokenizer
            # load json builder
            self.builder = ConstrainedGeneration(
                model=model,
                tokenizer=tokenizer,
                json_schema=json_schema,
                max_string_token_length=20)
        except:
            print("Model not found")
            return
        


    
    def generate(self,prompt):
        '''
        Given a prompt generate a new text wrt the json schema specified
        '''
        # Generate text

        output = self.builder(prompt)
        self.curr_output = output
        return output







    def feedback(self, feedback):
        '''
        If online learning is enabled, then get user feedback and 
        update the model online/zeroth order optimization

        TODO for v.0.1
        LORA hotswapping in the background since calculating the loss
        will add inference latency

        RN feedback is just the
        source of truth, but we can add more sophisticated feedback
        Online learning will do zeroth order optimization on delta(self.curr_output, feedback)

        '''
        # we know we have to go from generate -> feedback -> update
        if self.curr_output is None:
            print("No output to update")
            return
        else:
            # Implement the feedback mechanism here
            # This is a placeholder and needs to be replaced with actual implementation
            '''
            Lets run through how this is going to work
            1. We have a prompt and a json schema
            2. We generate a text
            3. We get feedback from the user/environment on the generated text and whether or not it is correct
            4. We update the model based on the feedback
            5. in this case feedback would be the correct output
            6. Send this to fwdpass learning w/ random perturbations to get estimated gradient and step
            7. Update the model
            8. Repeat
            sample format example

            sample = \
            Sample(
                id=example["idx"],
                data=example,
                candidates=[example["choice1"], example["choice2"]],
                correct_candidate=example[f"choice{example['label'] + 1}"],
            )
            <for now set it up in the format of >
            Prompt 
            Choices
            Correct choice
            example = {
            "idx": hash(sample),
            "data" = prompt
            "candidates" = choices
            "correct_candidate" = feedback
            }
            '''

            feedback.train(self.curr_output, feedback)
            

            
            

            
            
        
        



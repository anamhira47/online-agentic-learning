class Online:
    def __init__(self, json_schema, model_name):
        # json schema to constrain generation to 
        self.json_schema = json_schema
        # HF model to load
        self.model_name = model_name

        # Take care of loading for model through HF and tokenizer

    #
    def generate(self,prompt):
        '''
        Given a prompt generate a new text wrt the json schema specified
        '''


    def feedback(self, feedback):
        '''
        If online learning is enabled, then get user feedabck and 
        update the model online/zeroth order optimization

        TODO for v.0.1
        LORA hotswapping in the background since calculating the loss
        will add inference latency
        '''




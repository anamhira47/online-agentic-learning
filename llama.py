
model_id = './7B'
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
from constrainedGen import ConstrainedGeneration
from constrainedGen.format import highlight_values
# load peft adapter
#model = PeftModel.from_pretrained(model, '')

# raw inference




'''

with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=10)[0], skip_special_tokens=True))
    

'''

'''
prompt = "Generate a person's information based on the following schema:"
jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
generated_data = jsonformer()
'''
#print(generated_data)
'''
ecomm = {
    "type": "object",
    "properties": {
        "store": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "location": {"type": "string"},
                "inventory": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "productId": {"type": "string"},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "category": {"type": "string"},
                            "price": {"type": "number"},
                            "inStock": {"type": "boolean"},
                            "rating": {"type": "number"},
                            "images": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
        }
    },
}
'''

example_schema = {
    "type": "object",
    "properties": {
        "action": {"type": "string"}
    }
}



builder = ConstrainedGeneration(
    model=model,
    tokenizer=tokenizer,
    json_schema=example_schema,
    prompt="The camel is a large animal with a hump on its back, its thirsty, what action should it do",
    max_string_token_length=20,
)

print("Generating...")
output = builder()

highlight_values(output)
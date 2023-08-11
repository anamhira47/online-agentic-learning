# Sample data
from feedback import Framework, OurArguments
from tasks import get_task
from utils import encode_prompt, encode_samples
class Sample:
    def __init__(self, data):
        self.data = data

samples = [
    Sample({
        "sentence1": "The chicken is going to cross the road.",
        "sentence2": "I can't wait to cross the finish line.",
        "word": "cross"
    }),
    Sample({
        "sentence1": "He broke the window with a rock.",
        "sentence2": "He's trying to break the record.",
        "word": "break"
    }),
    Sample({
        "sentence1": "She will perform a song at the concert.",
        "sentence2": "The doctor will perform the surgery.",
        "word": "perform"
    }),
    Sample({
        "sentence1": "He will present his project to the class.",
        "sentence2": "This is a present for you.",
        "word": "present"
    }),
    Sample({
        "sentence1": "I left my keys on the table.",
        "sentence2": "He left the party early.",
        "word": "left"
    }),
    Sample({
        "sentence1": "She likes to play the piano.",
        "sentence2": "The children are playing in the park.",
        "word": "play"
    }),
    Sample({
        "sentence1": "He's going to check the mail.",
        "sentence2": "Can you check the answer for me?",
        "word": "check"
    }),
    Sample({
        "sentence1": "I need to charge my phone.",
        "sentence2": "The bull charged at the matador.",
        "word": "charge"
    }),
    Sample({
        "sentence1": "He's going to pass the ball.",
        "sentence2": "I hope I pass the test.",
        "word": "pass"
    }),
    Sample({
        "sentence1": "She's going to light the candle.",
        "sentence2": "It's light outside.",
        "word": "light"
    })
]

# Instantiate the template
#template = WICTemplate()

# Use the template to encode the sample
args = OurArguments()
print(args.task_name)

framework = Framework(args, get_task(args.task_name))

for sample in samples:
    encodings = encode_samples(framework.task.get_template(), [sample], framework.tokenizer, framework.args.max_length)
    # Create a possible label candidate of "No", "YES"
    label_candidates = ["No", "Yes"]
    
    for label_candidate in label_candidates:
        # Encode the label candidate
        encoded_label = framework.tokenizer.encode(label_candidate, add_special_tokens=False)
        # Add the encoded label to the label_id
        input_ids = encodings[0] + encoded_label
        print(f"Input ids for '{label_candidate}': {input_ids}")
        # Add the length of the encoded label
        label_len = len(encoded_label)
        
        logits = framework.forward(input_ids, option_len=label_len, generation=False)
        print(f"Logits for '{label_candidate}': {logits}")


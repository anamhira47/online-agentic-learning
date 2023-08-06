import unittest
import sys
sys.path.append('..')  # Add the parent directory to the Python path
from onlineAgent.online_learning import Online

class TestOnlineAgent(unittest.TestCase):

    def setUp(self):
        # Set up any necessary objects, connections, variables etc.
        self.json_schema = {
            "type": "object",
            "properties": {
                "number": {"type": "number"}
            }
        }

        self.model_name = 'meta-llama/Llama-2-7b-hf'  # Replace with your actual model name
        self.online = Online(self.json_schema, self.model_name)

    def test_online_init(self):
        # Test the initialization of the Online class
        self.assertIsInstance(self.online, Online)
        self.assertEqual(self.online.json_schema, self.json_schema)
        self.assertEqual(self.online.model_name, self.model_name)

    def test_generate(self):
        # Test the generate method
        prompt = "Generate a good number"  # Replace with your actual prompt
        output = self.online.generate(prompt)
        # Add assertions to check the output
        # For example, if output is a string, you can check if it's not empty
        print(output)
        self.assertIsInstance(output['number'], float)
        self.assertTrue(len(output) > 0)

    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()
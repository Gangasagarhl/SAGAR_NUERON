import torch 
from sagar_neuron_gpt2.load_model.load_local_models import LoadWeightsAndPrepareModel
from sagar_neuron_gpt2.processing.Text2TokenViceVersa import Converter
import tiktoken

class Inference:
    def __init__(self):
        """Initialize with device detection."""
        self.device = self._get_device()
        print(f"Using device: {self.device}")

    
    
    def _get_device(self):
        """Detect and return the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # For Apple Silicon Macs
            return torch.device("mps")
        else:
            return torch.device("cpu")




class LoadModel: 
    def __init__(self):
        print("Inference you are in\n")

    
    
    def take_config_from_file(self, path_to_config_file):
        import ast
        with open(path_to_config_file, 'r') as f:
            config_str = f.read()
        config_dict = ast.literal_eval(config_str)
        return config_dict
    


    def load_model(self):
        GPT_CONFIG = None

        path = input("Enter the path to config file:\n ")
        GPT_CONFIG = self.take_config_from_file(path)

      
        tokenizer = tiktoken.get_encoding("gpt2")
        conv = Converter()

        # Load model and optimizer
        model_weight_path = input("Cleanly enter the path to the model weights:  ")
        model = LoadWeightsAndPrepareModel().prepare_model(GPT_CONFIG, model_weight_path )
        model.eval()  # Set model to evaluation mode for inference

        # Determine model device
        device = next(model.parameters()).device
        return model, tokenizer, device




class Classifier:

    def __init__(self):
        modeling =  LoadModel()
        self.model, self.tokenizer, self.device = modeling.load_model()


    def classify(self,text, max_length=None, pad_token_id=50256):
        self.model.eval()

        input_ids = self.tokenizer.encode(text)
    

        supported_context_length = self.model.pos_emb.weight.shape[0]
        input_ids = input_ids[:min(max_length, supported_context_length)]

       
        input_ids += [pad_token_id] * (max_length - len(input_ids))
        input_tensor = torch.tensor(input_ids, device=self.device).unsqueeze(0) # add batch dimension

      
        with torch.no_grad():
            logits = self.model(input_tensor)[:, -1, :]  # Logits of the last output token
        predicted_label = torch.argmax(logits, dim=-1).item()

        # Return the classified result
        return "spam" if predicted_label == 1 else "not spam"


       
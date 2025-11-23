from sagar_neuron_gpt2.model_configuration.gpt_model import GPTModel
import torch

class LoadWeightsAndPrepareModel: 
    def __init__(self):
        print("Loading and preparing model")

    def prepare_model(self,MODELCONFIG, path):
        model =  GPTModel(MODELCONFIG)

        if  'num_classes' in MODELCONFIG:
            model.out_head = torch.nn.Linear(in_features=MODELCONFIG["emb_dim"], out_features=MODELCONFIG['num_classes'])

        checkpoint = torch.load(path, map_location='cpu' if not torch.cuda.is_available() else None)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.train()

        return model
    
    

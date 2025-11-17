from download_original_gpt2.main import GPT2Loader
from binary_classifier_using_gpt2.evaluate import Evaluate
from sagar_neuron_gpt2.TrainAndSaveGptWeights import EvaluateTrainModel

import torch
import tiktoken
import time


class GPT2Classifier:

    def __init__(self,path,train_loader=None, val_loader=None, test_loader=None):
        gpt2 = GPT2Loader()
        self.evaluate = Evaluate()
        self.save_info = EvaluateTrainModel()
        self.gpt2_model = gpt2.initialise()
        self.BASE_CONFIG = gpt2.configuaration()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.destination_path = path
        self.save_info.saving_model_configurataion(self.destination_path,self.BASE_CONFIG)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        


    def freeze_weights(self):
        for param in self.gpt2_model.parameters():
            param.requires_grad = False


        
    def classifer_head(self, number_of_transfomers_to_be_trained_from_end=1 ,num_classes=2):

        torch.manual_seed(123)
        self.gpt2_model.out_head = torch.nn.Linear(in_features=self.BASE_CONFIG["emb_dim"], out_features=num_classes)

        for param in self.gpt2_model.trf_blocks[number_of_transfomers_to_be_trained_from_end*-1].parameters():
            param.requires_grad = True

        for param in self.gpt2_model.final_norm.parameters():
            param.requires_grad = True



    #Check accuracy with data loaders
    def check_with_loader(self):

        model = self.gpt2_model
        device = self.device
        print("This is where it is failing\n")
        train_accuracy = self.evaluate.calc_accuracy_loader(self.train_loader, model, device)
        print("Train accuracy \n")
        val_accuracy = self.evaluate.calc_accuracy_loader(self.val_loader, model, device)
        print("va;idation accuracy\n")
        test_accuracy = self.evaluate.calc_accuracy_loader(self.test_loader,  model, device)
        print("Test  accuracy\n")



        print(f"Train accuracy: {train_accuracy*100:.2f}%")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        print(f"Test accuracy: {test_accuracy*100:.2f}%")




    #testing 2 nueron output
    def test(self, quesition="Do you have time"):

        inputs = self.tokenizer.encode(quesition)
        inputs = torch.tensor(inputs).unsqueeze(0)
        print("Inputs:", inputs)
        print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)

        with torch.no_grad():
            outputs = self.gpt2_model(inputs)

        print("Outputs:\n", outputs)
        print("Outputs dimensions:", outputs.shape) 



    #evaluate the model
    def evaluate_model(self, train_loader, val_loader, device, eval_iter):
        model = self.gpt2_model
        model.eval()
        with torch.no_grad():
            train_loss = self.evaluate.calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = self.evaluate.calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss
    


    #finetuning classfying core
    def train_classifier_simple(self, 
                                train_loader, 
                                val_loader, 
                                optimizer, 
                                device, 
                                num_epochs,
                                eval_freq, 
                                eval_iter):
        model = self.gpt2_model
        # Initialize lists to track losses and examples seen
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        examples_seen, global_step = 0, -1

        # Main training loop
        for epoch in range(num_epochs):

            model.train()  # Set model to training mode

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad() # Reset loss gradients from previous batch iteration
                loss = self.evaluate.calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward() # Calculate loss gradients
                optimizer.step() # Update model weights using loss gradients
                examples_seen += input_batch.shape[0] # New: track examples instead of tokens 
                global_step += 1

                ## 130 batches: training, eval_Freq = 50 --> after 50 batches are processed in each epoch, we print train loss and val loss

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model(
                         train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Calculate accuracy after each epoch
            train_accuracy = self.evaluate.calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
            val_accuracy = self.evaluate.calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
            print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
            print(f"Validation accuracy: {val_accuracy*100:.2f}%")
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)
            #save the model state and weights after each and every epoch
            self.save_info.saving_weights_of_model_optimizer(model,optimizer,epoch,self.destination_path,name_subfolder='finetuned_weights')


        return train_losses, val_losses, train_accs, val_accs, examples_seen
    

    def finetune(self, epochs):

        model = self.gpt2_model
        start_time = time.time()
        torch.manual_seed(123)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)    
        train_losses, val_losses, train_accs, val_accs, examples_seen = self.train_classifier_simple(
        
            self.train_loader, 
            self.val_loader, 
            optimizer, 
            self.device,
            num_epochs=epochs, 
            eval_freq=50, 
            eval_iter=5,
           
        )

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")







    

       
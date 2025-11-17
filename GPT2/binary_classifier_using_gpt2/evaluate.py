import torch
class Evaluate:

    def __init__(self):
        pass
   
    def calc_accuracy_loader(self,data_loader, model, device, num_batches=None):
        
        model.eval()
        correct_predictions, num_examples = 0, 0

        if num_batches is None:
            num_batches = len(data_loader)
            print("NUmber of batches: ", num_batches)

        else:
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                with torch.no_grad():
                    logits = model(input_batch)[:, -1, :]  # Logits of last output token
                predicted_labels = torch.argmax(logits, dim=-1)

                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break

        return correct_predictions / num_examples


    def calc_loss_batch(self,input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)[:, -1, :]  # Logits of last output token
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        return loss
    

    # Same as in chapter 5
    def calc_loss_loader(self,data_loader, model, device, num_batches=None):
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches
    


    

    

    

    

        
        
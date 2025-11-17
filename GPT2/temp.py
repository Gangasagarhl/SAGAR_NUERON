from binary_classifier_using_gpt2.finetuning_gpt import GPT2Classifier
from binary_classifier_using_gpt2.data_loaders  import Executer

if __name__ == "__main__":
    
    train_loader, val_loader,test_loader = Executer("binary_classifier_using_gpt2").execute()
    path = input("Give the path to store the weights and configuaration ")
    gpt2_classifier = GPT2Classifier(path,train_loader, val_loader, test_loader)
    gpt2_classifier.freeze_weights()
    gpt2_classifier.classifer_head(number_of_transfomers_to_be_trained_from_end=1, num_classes=2)
    gpt2_classifier.test("Do you have time")
    epo = int(input("Enter the number of epochs to finetune the model:  "))
    #gpt2_classifier.check_with_loader()
    gpt2_classifier.finetune(epochs=epo)




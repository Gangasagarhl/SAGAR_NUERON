from binary_classifier_using_gpt2.finetuning_gpt import GPT2Classifier
from binary_classifier_using_gpt2.data_loaders  import Executer
from binary_classifier_using_gpt2.inferencing_classifier import Classifier
import pyfiglet



class ClassifierGPT:
    
    def  __init__(self):
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.print_welcome()
        self.file_ready()
        

    def print_welcome(self):
        

        result = pyfiglet.figlet_format("WELCOME")
        print(result)

        

    def file_ready(self):
        tr_pth = input("Enter the path to training data csv file:  ")
        val_pth = input("Enter the path to validation data csv file:  ")
        te_pth = input("Enter the path to testing data csv file:  ")
        self.train_loader, self.val_loader,self.test_loader = Executer(train_path= tr_pth, test_path=te_pth, val_path=val_pth).execute()
        

        
        
    def training_script_normal_gpt(self):

        print("We are in training normal gpt2 model\n")
        
        
        path = input("Give the path to store the weights and configuaration ")
        number_of_transfomers_to_be_trained_from_end = int(input("Enter the number of transformer layers to be finetuned from the end:  "))
        num_classes = int(input("Enter the number of classes for classification:  ")) 
        epo = int(input("Enter the number of epochs to finetune the model:  "))
        sub_folder_name  =  input("Weights saving folder name: ")

        gpt2_classifier = GPT2Classifier(path,train_loader, val_loader, test_loader, number_of_transfomers_to_be_trained_from_end, num_classes, subfolder_name=sub_folder_name)
        gpt2_classifier.freeze_weights()

        gpt2_classifier.classifer_head(number_of_transfomers_to_be_trained_from_end,num_classes)
        gpt2_classifier.test("Do you have time")

        print("Before Finetuning\n")
        gpt2_classifier.check_with_loader()

        gpt2_classifier.finetune(epochs=epo)

        print("After Finetuning\n")
        gpt2_classifier.check_with_loader()



    def training_script_fourrier(self):
        print("Finetuning GPT2 model with fourier transformer\n")

        train_loader, val_loader,test_loader = Executer("binary_classifier_using_gpt2").execute()
        path = input("Give the path to store the weights and configuaration ")
        number_of_transfomers_to_be_trained_from_end = int(input("Enter the number of transformer layers to be finetuned from the end:  "))
        num_classes = int(input("Enter the number of classes for classification:  ")) 
        epo = int(input("Enter the number of epochs to finetune the model:  "))
        sub_folder_name  =  input("Weights saving folder name: ")
        ret = float(input("Enter the retain ratio for fourrier transform:  "))
        insert_every = int(input("Enter after how many layers the fourrier transform to be applied:  "))

        gpt2_classifier = GPT2Classifier(path,train_loader, 
                                        val_loader, 
                                        test_loader, 
                                        number_of_transfomers_to_be_trained_from_end, 
                                        num_classes, 
                                        subfolder_name=sub_folder_name,  
                                        retain_ratio=ret,
                                        insert_every=insert_every
                                        )
        
        gpt2_classifier.freeze_fourrier_weights()

        gpt2_classifier.fourier_classifier_head(number_of_transfomers_to_be_trained_from_end,num_classes)
        gpt2_classifier.test("Do you have time")
        

        print("Before Finetuning\n")
        gpt2_classifier.check_with_loader_fourier()

        gpt2_classifier.finetune_fourrier(epochs=epo)

        print("After Finetuning\n")
        gpt2_classifier.check_with_loader_fourier()




    def  inference_script(self):
        
        inferencing = Classifier()
        for i in range(5):
            text  = input("Enter the text to classify:  ")
            print(inferencing.classify(text))




    




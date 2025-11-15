def take_config_from_file( path_to_config_file):
        import ast
        with open(path_to_config_file, 'r') as f:
            config_str = f.read()
        config_dict = ast.literal_eval(config_str)
        return config_dict
    
if __name__ ==  "__main__": 
    entry =  take_config_from_file("./weights/config.txt")
    for k,v in entry.items():
        print(f"{k} : {v}")
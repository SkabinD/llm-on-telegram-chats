import os
import json
from datasets import Dataset
from typing import Dict, List
from tqdm import tqdm


def get_chats(config: Dict[str, str]) -> List[Dict]:
    data_path = os.path.join(config["DATA_DIR"], config["DATA_NAME"])
    
    with open(data_path, "r") as f:
        data = json.load(f)
        
    chats = data["chats"]["list"][1:]
    chats = sorted(chats, key=lambda x: len(x["messages"]), reverse=True)
    chats = chats[:20]
    
    return chats

def parse_data(chats: List[Dict], config: Dict[str, str]) -> List[Dict]:
    name = config["RESPONSE_NAME"]
    
    dialogs = []
    for chat in chats:
        messages = chat["messages"]
        
        is_me_first = True
        current_chat_name = chat["name"]
        message_sequence = []
        
        context, response = None, None
        
        if current_chat_name is None:
            current_chat_name = "Олег"
    
        for message in tqdm(messages):
            try: 
                name_from = message["from"]
            except KeyError:
                name_from = message["actor"]
            text = message["text"]
            
            if not text:
                continue
            
            if is_me_first:
                if name_from != name:
                    is_me_first = False
                else:
                    continue
                
            if isinstance(text, list):
                text = "\n".join(list(filter(lambda x: isinstance(x, str), text)))
            
            text = text.strip()
            
            if current_chat_name == name_from:
                pass
            else:
                output_text = "\n".join(message_sequence) + "\n\n"
                
                if current_chat_name == name:
                    response = output_text
                else: 
                    context = output_text
                    
                if all([context, response]):    
                    dialogs.append(
                            {
                                "context": context,
                                "response": response,
                                "chat": name_from
                            }
                        )
                    
                    context, response = None, None
                    
                message_sequence = []
                current_chat_name = name_from

            message_sequence.append(text)        
    return dialogs

def get_dataset(dialogs: List[Dict]) -> Dataset:
    return Dataset.from_list(dialogs)

def data_full_routine(config: Dict[str, str]) -> Dataset:
    chats = get_chats(config)
    dialogs = parse_data(chats, config)
    dataset = get_dataset(dialogs)
    return dataset

if __name__ == "__main__":
    import yaml 
    
    with open("configs/main.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    chats = get_chats(config)
    dialogs = parse_data(chats, config)
    print(dialogs[:5])

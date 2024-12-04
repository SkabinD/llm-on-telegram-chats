from datasets import Dataset
from typing import Dict, List
from tqdm import tqdm


def get_chats(raw_data: str, chat_sort_func=lambda x: len(x["messages"]), n_chats: int = 20) -> List[Dict]:
    chats = raw_data["chats"]["list"][1:]
    if chat_sort_func:
        chats = sorted(chats, key=chat_sort_func, reverse=True)
    chats = chats[:n_chats]
    return chats


def parse_data(chats: List[Dict], name: str) -> List[Dict]:
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


def data_full_routine(raw_data: Dict, name: str) -> Dataset:
    chats = get_chats(raw_data)
    dialogs = parse_data(chats, name)
    dataset = get_dataset(dialogs)
    return dataset


if __name__ == "__main__":
    import yaml 
    
    with open("configs/main.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    chats = get_chats(config)
    dialogs = parse_data(chats, config)
    print(dialogs[:5])

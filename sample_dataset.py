import json
import random
from pathlib import Path

def main():
    input_path = Path("data/hotpot_dev_distractor_v1.json")
    output_path = Path("data/hotpot_random_100.json")
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        return
        
    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} items. Sampling 100 items...")
    
    # Use a fixed seed for reproducibility
    random.seed(42)
    sampled_data = random.sample(data, 100)
    
    formatted_examples = []
    for item in sampled_data:
        # Extract fields
        qid = item.get("_id", "")
        difficulty = item.get("level", "medium")
        if difficulty not in ["easy", "medium", "hard"]:
            difficulty = "medium"
            
        question = item.get("question", "")
        gold_answer = item.get("answer", "")
        
        # Format context: HotpotQA context is list of lists [title, sentences]
        context_chunks = []
        raw_context = item.get("context", [])
        for chunk in raw_context:
            if isinstance(chunk, list) and len(chunk) >= 2:
                title = chunk[0]
                sentences = chunk[1]
                if isinstance(sentences, list):
                    text = " ".join(sentences)
                else:
                    text = str(sentences)
                context_chunks.append({
                    "title": title,
                    "text": text
                })
        
        formatted_examples.append({
            "qid": qid,
            "difficulty": difficulty,
            "question": question,
            "gold_answer": gold_answer,
            "context": context_chunks
        })
        
    print(f"Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_examples, f, indent=2, ensure_ascii=False)
        
    print("Done!")

if __name__ == "__main__":
    main()

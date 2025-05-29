from openai import OpenAI
import csv
import json
import argparse
import re


def find_video_by_name_csv(csv_path, video_name):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)

        try:
            name_idx = 0  # video_name default first row
            description_idx = headers.index("Description")
            analysis_idx = headers.index("Reasoning")
        except ValueError as e:
            print("Error: Required column not found in CSV headers.")
            return None, None

        for row in reader:
            if row[name_idx] == video_name:
                video_description = row[description_idx]
                video_analysis = row[analysis_idx]
                return video_description, video_analysis

    print(f"Video '{video_name}' not found in the CSV.")
    return None, None


def generate_answer(client, prompt, row):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an expert evaluator for video anomaly detection outputs."},
            {"role": "user", "content": prompt}
        ]
    )
    # print(response.choices[0].message.content)
    match = re.search(r"```json\n(.*?)\n```", response.choices[0].message.content, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json.loads(json_str)
    else:
        print("Json load error!")
        return None
    

def main(args):
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    existing_videos = {}
    # read the output json file for proceeded evaluation
    if args.json_output_path.endswith('.json'):
        with open(args.json_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                video_name = item.get("video_name")
                evaluation = item.get("evaluation", {})
                if video_name:
                    existing_videos[video_name] = evaluation

    total_scores = {
        "classification_score": 0,
        "key_matching_score": 0,
        "fluency_score": 0,
        "informativeness_score": 0,
        "factual_score": 0,
        "total_score": 0
    }
    count = 0

    with open(args.pred_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        total = len(rows)

    for idx, row in enumerate(rows):
        video_name = row['Video Name']
        if video_name in existing_videos.keys():
            print(f"Found existing evaluation for video: {video_name}, skipping call to DeepSeek API.")
            total_scores = {key: total_scores[key] + existing_videos[video_name].get(key, 0) for key in total_scores}
            count += 1
            print("Current average scores:")
            for key in total_scores:
                avg = total_scores[key] / count
                print(f"  {key}: {avg:.2f}")
            continue    


        description = row["Description"]
        analysis = row["Analysis"]
        gt_description, gt_analysis = find_video_by_name_csv(args.test_gt_path, video_name)

        evaluation_prompt = f"""
        Below is a ground-truth description and analysis, followed by a model-generated description and analysis. Please evaluate the model's outputs from the following aspects:

        1. Classification Correctness (10 pts)  
        2. Key Object and Action Matching (10 pts)  
        3. Fluency and Coherence (10 pts)  
        4. Informativeness and Domain Awareness (10 pts)  
        5. Factual Consistency (10 pts)

        ---
        Ground Truth:
        - Description: {gt_description}
        - Analysis: {gt_analysis}

        Model Output:
        - Description: {description}
        - Analysis: {analysis}

        Please return a JSON object strictly formatted as:

        {{
            "classification_score": int (0-10),
            "key_matching_score": int (0-10),
            "fluency_score": int (0-10),
            "informativeness_score": int (0-10),
            "factual_score": int (0-10),
            "total_score": int (0-50),
            "comments": {{
                "classification": "...",
                "key_matching": "...",
                "fluency": "...",
                "informativeness": "...",
                "factual": "..."
            }}
        }}
        """

        if gt_description and gt_analysis and description and analysis:
            answer = generate_answer(client, evaluation_prompt, row)

            result = {
                "video_name": video_name,
                "evaluation": answer
            }

            with open(args.json_output_path, 'a', encoding='utf-8') as out_json:
                json.dump(result, out_json, indent=4, ensure_ascii=False)
            
            count += 1
            for key in total_scores:
                total_scores[key] += answer.get(key, 0)

            print("Current average scores:")
            for key in total_scores:
                avg = total_scores[key] / count
                print(f"  {key}: {avg:.2f}")
        else:
            print("The input is not complete.")

    print("\nFinal average scores:")
    for key in total_scores:
        avg = total_scores[key] / count if count > 0 else 0
        print(f"  {key}: {avg:.2f}")
    print(f"Evaluation results saved to {self.json_output_path}\n", "-"*40, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate video anomaly detection output using DeepSeek.")
    parser.add_argument('--pred_path', type=str, required=True, help="Path to the CSV file containing model predictions.")
    parser.add_argument('--test_gt_path', type=str, required=True, help="Path to the CSV file containing ground truth data for evaluation.")
    parser.add_argument('--api_key', type=str, required=True, help="API key for DeepSeek.")
    parser.add_argument('--base_url', type=str, default="https://api.deepseek.com", help="Base URL for DeepSeek API.")
    parser.add_argument('--json_output_path', type=str, default=None, help="Path to save the evaluation results in JSON format.")
    args = parser.parse_args()

    if args.json_output_path is None:
        args.json_output_path = args.pred_path.replace('.csv', '_eval.json')
    main(args)
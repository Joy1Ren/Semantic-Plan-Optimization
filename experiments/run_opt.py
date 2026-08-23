from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
import json
import pandas as pd
import time

from agent_cost_model.cost_model_agent import CostModelAgent, OpenRouterClient, ResultsStore
from agent_cost_model.llm_sampler import LLM_Sampler
from agent_cost_model.paths import DATASET_DIR, RESULTS_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use the scripted LLM (no API key)")
    ap.add_argument("--model", default="openai/gpt-5.4", help="OpenRouter model id")
    ap.add_argument("--helper-model", default="openai/gpt-5.4", help="OpenRouter model id for the cost helper agent (customCost mode); defaults to --model")
    ap.add_argument("--max-steps", type=int, default=40, help="max steps for the main agent (cost helper has its own max)")
    ap.add_argument("--runcount", type=int, default=None, help="run index; saves result under key '<query_id>_<runcount>' instead of overwriting '<query_id>'")
    ap.add_argument("--embedding-model", default=None, help="embedding model id for LLM_Sampler; when set, the quality-eval subset is drawn by LLM-guided sampling instead of at random. Which modalities are embedded is read from the `modalities` map below; use a multimodal model for queries whose modality includes 'image'.")
    ap.add_argument("--sample-method", choices=["importance-sampling", "topk"], default="importance-sampling", help="how LLM_Sampler picks rows from cosine-similarity scores: 'importance-sampling' (weighted random draw) or 'topk' (highest-scoring rows)")
    ap.add_argument("--keyword", action="store_true", help="add +1 to a row's similarity if any full word of the LLM target text appears in the row's text data (full-word match)")
    ap.add_argument("--no_image_emb", action="store_true", help="ignore image embeddings when scoring, even if the dataset has images (avoids costly image embedding)")
    args = ap.parse_args()

    plans = {}
    op_results = ResultsStore([])
    plan_results = ResultsStore([])

    agent_type = "execute_oracle_sampler"
    # agent_type = "customCost_oracle_helper"
    USE_CASE = "cuad"
    SCALE_FACTOR = 16000
    FINAL_EVAL_RUNS = 2
    llm = ScriptedLLM() if args.offline else OpenRouterClient(args.model, reasoning_effort="medium")
    agent = CostModelAgent(llm, max_steps=args.max_steps, verbose=True, agent_dir=f"{agent_type}_agent", use_case = USE_CASE, helper_model=args.helper_model)


    query_id = 10
    source_data = f"Reviews_{SCALE_FACTOR}.csv" #just for LLM_Sampler, not directly given to agent
    data_dir = str(DATASET_DIR / USE_CASE / f"sf_{SCALE_FACTOR}")
    # Per-query modality: drives whether the sampler embeds/compares text, images, or both.
    modalities = {
        "ecomm":{
            **{q: "image and text" for q in [8, 10, 11, 12, 13, 14]},
            **{q: "text" for q in [1, 3, 5, 7]},
            **{q: "image" for q in [2, 4, 6, 9]},
        },
        "movie": {
            **{q: "text" for q in range(1,11)}
        }
    }
    eval_metrics = {
        "ecomm": {
            1: "f1-score", 2: "f1-score", 3: "adjusted-rand-index", 4: "adjusted-rand-index",
            5: "adjusted-rand-index", 6: "adjusted-rand-index", 7: "f1-score", 8: "f1-score",
            9: "f1-score", 10: "f1-score", 11: "f1-score", 12: "f1-score", 13: "f1-score",
            14: "f1-score",
        },
        "movie": {
            3: "relative-error", 7: "f1-score", 9: "spearman-rank", 10 : "spearman-ranks"
        }
    }
    tasks = {
        "ecomm":{
            1: """Based on the textual description and the title of the product,
            find the `prod_id` of products that are backpacks from Reebok.""",
            2: """Based on the image representation of the product,
                find the `prod_id` of products where the image shows
                a pair of sports shoes that are predominantly yellow and silver.""",
            3: """For each product, extract the brand name from the product description and title.
                Return the product id and the brand in a column titled 'category'.""",
            4: """For each product, use the image to extract the single primary color of the depicted product.
                Return `prod_id` and the primary color in a column titled 'category'.""", #note: added "single"
            5: """Based solely on the title and description of the product, classify each product into one of the following categories:
                Dress: A dress is a one-piece outer garment that is worn on the torso, hangs down over the legs, and often consist of a bodice attached to a skirt.
                Bottomwear: Bottomwear refers to clothing worn on the lower part of the body, such as trousers, jeans, skirts, shorts, and leggings.
                Socks: Socks are a type of clothing worn on the feet, typically made of soft fabric, designed to provide comfort and warmth.
                Topwear: Topwear refers to clothing worn on the upper part of the body, such as shirts, blouses, t-shirts, and jackets.
                Innerwear: Innerwear refers to clothing worn beneath outer garments, typically close to the skin, such as underwear, bras, and undershirts.
                Each product can only have one category.
                Return `prod_id` and the category.""",
            6: """Based solely on the image of the product, classify each product into one of the following categories:
                Dress: A dress is a one-piece outer garment that is worn on the torso, hangs down over the legs, and often consist of a bodice attached to a skirt.
                Bottomwear: Bottomwear refers to clothing worn on the lower part of the body, such as trousers, jeans, skirts, shorts, and leggings.
                Socks: Socks are a type of clothing worn on the feet, typically made of soft fabric, designed to provide comfort and warmth.
                Topwear: Topwear refers to clothing worn on the upper part of the body, such as shirts, blouses, t-shirts, and jackets.
                Innerwear: Innerwear refers to clothing worn beneath outer garments, typically close to the skin, such as underwear, bras, and undershirts.
                Each product can only have one category.
                Return `prod_id` and the category.""",
            7: """Find all pairs of products priced at $500 or less where both products
                are of the same category and from the same brand based on their descriptions.
                The pairs may be the same and output all orders of pairs (e.g., include pair a,b and b,a and a,a).
                Return the two `prod_id` columns in this order: first product's `prod_id`, second product's `prod_id`.""", ## added instruction to not remove pairs
            8: """Perform a self-join of the dataset.
                For each product, find the matching product images based on the title
                and the description of the product. You MAY NOT use the `prod_id` to make the join.
                You must use the image to match the text.""", #removed 3000 min character
            9: """Based on product images, find pairs of distinct products under $800
                in a single base color (Black, Blue, Red, White, Orange, or Green)
                that depict objects of the same category and the same dominant surface color.
                The pairs MAY NOT be the same but output all order of pairs (e.g., include pair a,b and b,a but not a,a).
                Return the two `prod_id` columns in this order: first product's `prod_id`, second product's `prod_id`.""",
            10: """Based on product images and descriptions, find matching outfits consisting of shoes,
                bottomwear, and topwear in Black, Blue, Red, or White,
                where all three items are from the same brand, same color,
                and each is priced at $1000 or less. Return the three `prod_id` columns in
                this order: shoes `prod_id`, bottomwear `prod_id`, topwear `prod_id`.""", #added "descriptions" because pz and lotus use text columns too
            11: """Based on product images and descriptions, find matching all-black outfits
                consisting of shoes, bottomwear (excluding swimwear), topwear (excluding swimwear),
                and an accessory (watch, jewellery, or bag priced at $500 or less),
                where all four items are from the same brand. Return the four prod_id columns
                in this order: shoes prod_id, bottomwear prod_id, topwear prod_id, accessory prod_id""",
            12: """For each Adidas or Puma product, use the product image and description
                to generate the following columns in this order: prod_id, brand name (lowercase),
                and master category classified as 'accessories', 'apparel', or 'footwear'.""",
            13: """Based on product images and descriptions, find men's running shirts
                with round neck and short sleeves, in blue or black (not bright colors
                like white, and definitely not green), with a striped design,
                suitable for outdoor running in warm weather. Return prod_id.""",
            14: """For each fashion product that costs less than 130,
                find the single image that best matches the product's textual description
                (including the product name and full description),
                but only if that image also depicts white socks."""
        },
        "movie":{
            3: """Count of positive reviews for movie `taken_3`. Return one column: `positive_review_cnt`.""",
            7: """All Pairs of reviews that express the *opposite* sentiment (one is positive and the other is negative) for movie with id `ant_man_and_the_wasp_quantumania`.
                The two reviewIds MAY NOT be the same but output both orders of pairs (e.g., include pair a,b and b,a but not a,a).
                Return columns in this order: `movie_name`, `reviewId1`, `reviewId2`""",
            9: """Score from 1 to 5 how much did the reviewer like the movie based on the movie reviews for movie `ant_man_and_the_wasp_quantumania`.
                Return columns in this order: `reviewId`, `reviewScore`.""",
            10: """Rank the movies based on movie reviews. For each movie, score every review of it from 1 to 5, then calculate the average score of these reviews for each movie.
                "Return columns in this order: `movieId`, `movieScore`.""",
        }
    }
    task = tasks[USE_CASE][query_id]

    # LLM-guided sampling of the quality-eval subset
    if args.embedding_model:
        source_df = pd.read_csv(os.path.join(data_dir, source_data), dtype={"idx": str})
        modality = modalities[USE_CASE][query_id]
        image_dir = os.path.join(data_dir, "images") if "image" in modality else None
        cache_dir = str(RESULTS_DIR / "sampling" / USE_CASE)
        text_cols = None if "text" in modality else []
        sampler = LLM_Sampler(
            query_id=query_id,
            use_case=USE_CASE,
            scale_factor=SCALE_FACTOR,
            query_text=task,
            df=source_df,
            llm_client=llm,
            embedding_model=args.embedding_model,
            image_dir=image_dir,
            cache_dir=cache_dir,
            text_cols=text_cols,
            id_col="reviewId",
            sample_method=args.sample_method,
            keyword=args.keyword,
            no_image_emb=args.no_image_emb,
        )
        sampled_ids = sampler.sample()
        print(f"\n=== LLM_Sampler subset ({len(sampled_ids)} rows) -> {sampler.subset_out_path} ===")
        print(sampled_ids)

    answer = agent.run(task, plans,
                       plan_results = plan_results,
                       op_results = op_results,
                       mode = agent_type,
                       query_info={
                            "use_case": USE_CASE,
                            "scale_factor": SCALE_FACTOR,
                            "query_id": query_id,
                            "eval_metric": eval_metrics[USE_CASE][query_id],
                            "final_eval_runs": FINAL_EVAL_RUNS,
                            "data_dir": data_dir,
                            "gt_dir": f"files/{USE_CASE}/raw_results/ground_truth/sf_{SCALE_FACTOR}",
                            "image_id_col": "idx",
                            "image_subdir": "images",
                            "image_ext": ".jpg",
                        "runcount": args.runcount,
                        })

    print("\n=== FINAL ANSWER ===")
    print(answer)
    output_path = RESULTS_DIR / "final_answer" / USE_CASE / f"{agent_type}_agent" / f"sf_{SCALE_FACTOR}" / f"Q{query_id}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    run_key = str(args.runcount) if args.runcount is not None else "0"
    answers_by_run = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            answers_by_run = json.load(f)
    answers_by_run[run_key] = answer
    with open(output_path, "w") as f:
        json.dump(answers_by_run, f, indent=2)
    print("\n=== observed-results store after run ===")
    print(op_results.summary())


if __name__ == "__main__":
    main()

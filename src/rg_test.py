from utils.rouge_score import evaluate_rouge, evaluate_rouge_avg

evaluate_rouge_avg(["hello, this is Adam Sterling from Arlington", "My mother name is Anna. I was born in Seattle, WA."],
               ["My name is Adam Sterling, a human being living in Arlington, VA", "Anna is my mother, love her so much. We both live in Seattle, WA"])
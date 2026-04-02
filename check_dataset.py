import pyarrow.parquet as pq
import json

datasets = {
    "MATH":  "d:/Desktop/work/verl0.7.0/data/math/test.parquet",
    "GSM8K": "d:/Desktop/work/verl0.7.0/data/gsm8k/test.parquet",
}

for name, path in datasets.items():
    print(f"=== {name} ===")
    m = pq.read_metadata(path)
    print(f"  rows: {m.num_rows}  created_by: {m.created_by}")
    rg = m.row_group(0)
    print(f"  columns ({rg.num_columns}):")
    for i in range(rg.num_columns):
        c = rg.column(i)
        print(f"    {c.path_in_schema}: {c.physical_type}")

    # Try reading a few rows with pandas
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        print(f"  pandas columns: {list(df.columns)}")
        row = df.iloc[0]
        print(f"  sample reward_model: {row.get('reward_model', 'N/A')}")
        print(f"  sample prompt: {str(row.get('prompt', 'N/A'))[:150]}")
        print(f"  sample data_source: {row.get('data_source', 'N/A')}")

        # Check if instruction suffix is in prompt
        prompt = row.get('prompt', [])
        if isinstance(prompt, list) and len(prompt) > 0:
            content = prompt[0].get('content', '') if isinstance(prompt[0], dict) else str(prompt[0])
            has_boxed = 'boxed' in content.lower()
            has_stepbystep = 'step by step' in content.lower()
            print(f"  prompt has 'boxed': {has_boxed}")
            print(f"  prompt has 'step by step': {has_stepbystep}")

    except Exception as e:
        print(f"  pandas error: {e}")
    print()

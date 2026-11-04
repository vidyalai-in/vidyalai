import json, collections, pandas as pd
rows=[json.loads(l) for l in open("data/votes-140k.jsonl")]
df=pd.DataFrame(rows)
print(len(df)); print(df.winner.value_counts())
print(df.timestamp.min(), df.timestamp.max())
print(df.language.value_counts().head(8))
print(df.is_code.value_counts())
ms=pd.concat([df.model_a,df.model_b]).value_counts()
print(len(ms)); print(ms.to_string())
print(json.dumps(rows[0]["conv_metadata"])); print(json.dumps(rows[0]["category_tag"]))
print(df.evaluation_order.value_counts().head())

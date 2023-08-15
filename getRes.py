import glob
import pandas as pd

from src.utils.vis_utils import get_df, average_df
exp_mode=f"mocov3-attention-grid"

LOG_NAME = "logs.txt"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

root = f"/share/ckpt/cgn/vpt/output-{exp_mode}"
df_list=[] 
for seed in ["40", "42", "314", "511", "666","800","2023" ,"13", "25", "4647", "197" ,"768", "314100"]:
#     model_type = f"adapter_{r}"
    files = glob.glob(f"{root}/seed{seed}/*/*/*/run1/{LOG_NAME}")
    for f in files:
        df = get_df(files, f"seed{seed}", root, is_best=True, is_last=True)
        if df is None:
            continue
        df["seed"] = seed
    df_list.append(df)


df = pd.concat(df_list).drop_duplicates()

df["type"] = f"{exp_mode}"

with open(f"{exp_mode}-display.csv",'w') as res:
        res.write(df.to_csv())
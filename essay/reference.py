import pandas as pd

# 读取CSV
df = pd.read_csv("export2025.09.24-01.47.34.csv")

# 函数：格式化单条引文
def format_ieee_reference(row, index):
    authors = row["Authors"].replace(";", ",")
    title = row["Document Title"]
    pub_title = row["Publication Title"]
    year = row["Publication Year"]
    volume = row["Volume"] if not pd.isna(row["Volume"]) else ""
    issue = row["Issue"] if not pd.isna(row["Issue"]) else ""
    start = str(row["Start Page"]) if not pd.isna(row["Start Page"]) else ""
    end = str(row["End Page"]) if not pd.isna(row["End Page"]) else ""
    doi = row["DOI"] if not pd.isna(row["DOI"]) else ""
    
    # 拼接引文
    citation = f"[{index+1}] {authors}, \"{title},\" {pub_title}, "
    if volume:
        citation += f"vol. {volume}, "
    if issue:
        citation += f"no. {issue}, "
    if start and end:
        citation += f"pp. {start}-{end}, "
    citation += f"{year}."
    if doi:
        citation += f" doi: {doi}"
    return citation

# 生成参考文献列表
references = [format_ieee_reference(row, i) for i, row in df.iterrows()]

# 输出到文本文件
with open("references.txt", "w", encoding="utf-8") as f:
    for ref in references:
        f.write(ref + "\n")

print("已生成 references.txt，内容如下：")
print("\n".join(references[:5]))  # 只打印前5条做示例

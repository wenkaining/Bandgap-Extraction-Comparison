import pandas as pd
import re

def clean_and_normalize(xlsx_path, comparison_xlsx, code, sheet_name=0):
    """清洗和标准化数据
    从 doi, material, value(with unit) 开始的统一 post-processing
    
    Args:
        xlsx_path: 结果Excel - "FINAL_{CODE}_{dtime}.xlsx"
        sheet_name: 要处理的sheet - doi, material, value(with unit)
    """
    # 0. 基础清洗
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # 去空格
    # 替换各种形式的连字符为标准减号
    special_dashes = ["−", "–", "—", "‐", "‑", "⁃", "₋", "⁻"]
    for dash in special_dashes:
        df["material"] = df["material"].str.replace(dash, "-")
    df = df.drop_duplicates()  # 去重
    df["doi"] = df["doi"].str.replace("_", "/")  # 统一 doi
    df["del"] = False  # 添加 del

    # 1. 分离value和unit（得到 doi, material, value, unit）
    def process_value(x):
        if pd.isna(x):
            return "", "", True
        x = str(x)
        if "meV" in x:
            return x.replace("meV", "").strip(), "meV", False
        elif "eV" in x:
            return x.replace("eV", "").strip(), "eV", False
        return x.strip(), "", True
    if "unit" not in df.columns:
        df["value"], df["unit"], df["del"] = zip(*df["value"].apply(process_value))

    # 2. 数据筛选（得到 doi, material, value, unit, del）
    # 删除1：单位unit未包含 eV 或 meV
    df.loc[~df["unit"].str.contains("eV", case=False, na=False), "del"] = True
    # 删除 2：特定关键词的材料
    keywords = ['brookite', 'Brookite', 'nitrogen', 'Nitrogen', 'oxygen', 'Oxygen', 
               'VB', 'VBM', 'TM', 'transition metal', 'VOC']
    df.loc[df["material"].isin(keywords), "del"] = True
    # 删除3：以+/-结尾的材料名
    df.loc[df["material"].str[-1].isin(["+", "-"]), "del"] = True
    # 删除4：不在allowed_names列表中的短材料名
    allowed_names = ["Ge", "Sn", "Si", "Se", "Te", "C", "S"]
    df.loc[(df['material'].str.len().isin([1, 2])) & (~df['material'].isin(allowed_names)), 'del'] = True
    # 删除5：不包含数字的value
    df.loc[~df["value"].apply(lambda x: bool(re.search(r"\d", str(x)))), "del"] = True
    # # 删除条件6：value列中的数字范围不在0~20之间
    # def extract_numbers(value):
    #     # 修改这个函数来正确处理范围值
    #     if "-" in value and not value.startswith("-"):
    #         # 如果有减号，但不是以减号开头，那么这是一个范围
    #         numbers = value.split("-")
    #         return [float(num.strip()) for num in numbers]
    #     return [float(num) for num in re.findall(r"-?\d+\.?\d*", str(value))]
    # def is_valid_range(numbers):
    #     return all(0 <= num <= 20 for num in numbers)
    # df["extracted_numbers"] = df["value"].apply(extract_numbers)
    # df.loc[~df["extracted_numbers"].apply(is_valid_range), "del"] = True
    # df = df.drop("extracted_numbers", axis=1)
    
    # # 删除7：尝试将value列中每一个内容转为数字，如果不行，且其中不包含"to","~","-"...，则删去
    # df.loc[~df["value"].apply(lambda x: pd.to_numeric(x, errors='coerce')).notnull() & 
    #        ~df["value"].str.contains("to|~|-|<|>", case=False, na=False), "del"] = True
    
    # 排序
    new_order = ["doi", "material", "value", "unit", "del"]
    new_order += [col for col in df.columns if col not in new_order]
    df = df[new_order]
    with pd.ExcelWriter(xlsx_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name="2-clean", index=False)
    # 删去
    df_clean = df[~df["del"]].drop(columns=["del"])
    print(f"初步筛选删去：{len(df[df['del']]) }/{len(df)}")

    # 3. value列数值标准化（得到 doi, material, value, error, sentence）
    """注意：后续还要处理range的数据"""
    df_clean["error"] = ""
    def process_value_format(value):
        """处理value列中的数字格式"""
        # 确保value是字符串类型
        value = str(value)
        # 替换特殊字符为标准字符
        replacements = {"−": "-", "–": "-", "—": "-", "～": "~", "∼": "~", "≈": "~"}
        for old, new in replacements.items():
            value = value.replace(old, new)
        # 移除数字内部的空格
        value = re.sub(r"(\d)\s+(\.|,)\s+(\d)", r"\1\2\3", value)
        # 在数字和字母之间添加一个空格
        value = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", value)
        value = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", value)
        # 处理特殊情况，如"3.43to3.22"->"3.43 to 3.22"，确保有且只有一个空格
        value = re.sub(r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)", r"\1 to \2", value)
        # 移除多余的空格
        value = re.sub(r"\s+", " ", value).strip()
        return value
    df_clean["value"] = df_clean["value"].apply(process_value_format)

    # 将meV转换为eV
    def convert_unit(value, unit):
        """转换数值并根据单位调整精度，保留原格式中的非数字部分。"""
        if unit in ("m", "M", "k", "K"):
            return value, unit

        value = str(value)
        parts = re.split(r"([-+]?\d+\.?\d*)", value)
        matches = re.findall(r"[-+]?\d+\.?\d*", value)

        if not matches:
            return value, unit

        factor = None
        new_unit = unit
        if (unit.startswith("k") or unit.startswith("K")) and len(unit) > 1:
            factor = 1000
            new_unit = unit[1:]
        elif (unit.startswith("m") or unit.startswith("M")) and len(unit) > 1:
            factor = 0.001
            new_unit = unit[1:]
        else:
            return value, unit

        converted = []
        for num_str in matches:
            if "." in num_str:
                _, decimal_part = num_str.split(".", 1)
                original_places = len(decimal_part)
            else:
                original_places = 0

            num = float(num_str) * factor
            if unit.startswith("k"):
                decimal_places = max(original_places - 3, 0)
            else:
                decimal_places = original_places + 3

            if decimal_places <= 0:
                formatted = f"{int(round(num))}"
            else:
                formatted = f"{num:.{decimal_places}f}"
            converted.append(formatted)

        converted_parts = []
        idx = 0
        for part in parts:
            if re.fullmatch(r"[-+]?\d+\.?\d*", part):
                if idx < len(converted):
                    converted_parts.append(converted[idx])
                    idx += 1
                else:
                    converted_parts.append(part)
            else:
                converted_parts.append(part)

        return "".join(converted_parts), new_unit
    df_clean[["value", "unit"]] = df_clean.apply(
        lambda row: pd.Series(convert_unit(row["value"], row["unit"])), axis=1
    )
    # df_clean = df_clean.drop(columns=["unit"])
    
    def split_pm_value(value):
        """处理包含 ± 的数据"""
        if isinstance(value, str) and "±" in value:
            main_val, error_val = value.split("±")
            return main_val.strip(), error_val.strip()
        return value, None
    
    # 应用分割函数并更新DataFrame
    for idx, row in df_clean.iterrows():
        if isinstance(row["value"], str) and "±" in row["value"]:
            main_val, error_val = split_pm_value(row["value"])
            df_clean.at[idx, "value"] = main_val
            df_clean.at[idx, "error"] = error_val
    
    # 删除value列为空的行
    original_count = len(df_clean)  # 记录删除前的行数
    df_clean.dropna(subset=['value'], inplace=True)
    deleted_rows = original_count - len(df_clean)  # 计算被删除的行数
    print(f"已删去value包含空值的行数: {deleted_rows}")
    
    """后续可能要删除（这里是不管句子不同，统一篇文章出现多次该数据条目，只保留第一条）"""
    # 根据"doi", "material", "value", "error"列去除重复行，保留第一行
    original_count = len(df_clean)  # 记录删除前的行数
    df_clean.drop_duplicates(subset=["doi", "material", "value", "error"], keep='first', inplace=True)
    deleted_rows = original_count - len(df_clean)  # 计算被删除的行数
    print(f"已删去（不管sentence不一样的）重复的行数: {deleted_rows}")
    
    # 重新排序
    new_order = ["doi", "material", "value", "error"]
    new_order += [col for col in df_clean.columns if col not in new_order]
    df_clean = df_clean[new_order]
    
    # 保存结果
    with pd.ExcelWriter(xlsx_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        df_clean.to_excel(writer, sheet_name="3-normalize", index=False)
    with pd.ExcelWriter(comparison_xlsx, mode="a", if_sheet_exists="replace", engine="openpyxl") as writer:
        df_clean.to_excel(writer, sheet_name=f"{code}_clean", index=False)

def compare(xlsx_path, comparison_xlsx, code):
    """比较提取结果与标准答案
    
    Args:
        xlsx_path: 结果Excel - "FINAL_{CODE}_{dtime}.xlsx"
        comparison_xlsx: 标准答案文件路径
        code: 提取方法代码
    """
    error_column = code + "_error"  # 误差列名
    gt_df = pd.read_excel(comparison_xlsx, sheet_name="summary")
    toeval_df = pd.read_excel(xlsx_path, sheet_name=-1)
    toeval_df["match"] = ""  # 添加新列 "match"
    # 将value列转为字符串类型
    gt_df["value"] = gt_df["value"].astype(str)
    toeval_df["value"] = toeval_df["value"].astype(str)
    
    # 保存原始material列
    toeval_df["original_material"] = toeval_df["material"].copy()
    # 将material列转为小写
    toeval_df.loc[:, "material"] = toeval_df["material"].apply(lambda x: x.lower() if isinstance(x, str) else x)
    
    match_count = 0
    for idx, gt_row in gt_df.iterrows():
        comp_material = gt_row['material']
        # 处理material匹配：comp_material 可能有 ";" 分割的多个值
        comp_materials = [m.strip().lower() if isinstance(m.strip(), str) else m.strip()
                         for m in comp_material.split(';')] if isinstance(comp_material, str) else [comp_material]
        
        # 尝试将value转为数字进行比较
        gt_value = gt_row["value"]
        try:
            gt_value_num = float(gt_value)
            value_comparison = toeval_df["value"].apply(lambda x: float(x) if isinstance(x, str) and (x.replace('-','',1).replace('.','',1).isdigit()) else x) == gt_value_num
        except ValueError:
            value_comparison = toeval_df["value"] == gt_value
            
        match = toeval_df[
            (toeval_df["doi"] == gt_row["doi"]) &
            (toeval_df["material"].isin(comp_materials)) &
            value_comparison &
            ((toeval_df["error"] == gt_row["error"]) | 
             (pd.isna(toeval_df["error"]) & pd.isna(gt_row["error"])))
        ]
        if not match.empty:
            gt_df.at[idx, code] = match.iloc[0]["value"]
            gt_df.at[idx, error_column] = match.iloc[0]["error"]
            match_count += 1
            toeval_df.loc[match.index, "match"] = 1  # 在匹配成功的行的"match"列填入1
    
    """后续可删"""
    """在“toeval_df与gt_df比较”后、“gt_df末尾补充条目”之前，添加内容：
如果gt_df有"other_mat"列，则将上一步比较后剩余的toeval_df内容与gt_df进行与上面一样的比较（用"other_mat"列代替"material"列，其余比较不变）、相同的在对应列填入数值，不同的是不需要在match列填入1。“gt_df末尾补充条目”步骤中，"other_mat"列不再重复添加这些已经匹配上的"""
    # 如果gt_df有"other_mat"列，则用other_mat列再次比较未匹配的数据
    if "other_mat" in gt_df.columns:
        # 找出未匹配的行
        unmatched_df = toeval_df[toeval_df["match"] != 1]
        
        for idx, gt_row in gt_df.iterrows():
            if pd.notna(gt_row["other_mat"]):
                comp_other_mat = gt_row['other_mat']
                # 处理other_mat匹配：comp_other_mat 可能有 ";" 分割的多个值
                comp_other_mats = [m.strip().lower() if isinstance(m.strip(), str) else m.strip()
                                for m in comp_other_mat.split(';')] if isinstance(comp_other_mat, str) else [comp_other_mat]
                
                # 尝试将value转为数字进行比较
                gt_value = gt_row["value"]
                try:
                    gt_value_num = float(gt_value)
                    value_comparison = unmatched_df["value"].apply(lambda x: float(x) if isinstance(x, str) and (x.replace('-','',1).replace('.','',1).isdigit()) else x) == gt_value_num
                except ValueError:
                    value_comparison = unmatched_df["value"] == gt_value
                    
                match = unmatched_df[
                    (unmatched_df["doi"] == gt_row["doi"]) &
                    (unmatched_df["material"].isin(comp_other_mats)) &
                    value_comparison &
                    ((unmatched_df["error"] == gt_row["error"]) | 
                    (pd.isna(unmatched_df["error"]) & pd.isna(gt_row["error"])))
                ]
                if not match.empty:
                    gt_df.at[idx, code] = match.iloc[0]["value"]
                    gt_df.at[idx, error_column] = match.iloc[0]["error"]
                    match_count += 1
                    toeval_df.loc[match.index, "match"] = 2  # 在匹配成功的行的"match"列填入2
    """添加：将没有match的toeval_df内容追加入gt_df末尾，doi和material与原来gt_df的对应，value和error的内容同时填入value、error以及code、error_column，相同的内容在同一行中"""
    # 找出没有匹配的行
    unmatched_df = toeval_df[(toeval_df["match"] != 1) & (toeval_df["match"] != 2)]
    # 在gt_df中添加other_mat列在material列之前
    if "other_mat" not in gt_df.columns:
        material_idx = gt_df.columns.get_loc("material")
        gt_df.insert(material_idx, "other_mat", "")
    # 为每个未匹配的行创建新行添加到gt_df
    for _, row in unmatched_df.iterrows():
        new_row = pd.Series(index=gt_df.columns)
        new_row["doi"] = row["doi"]
        new_row["other_mat"] = row["original_material"]  # 使用原始大写的material
        # new_row["material"] = row["material"]  # 使用小写的material
        new_row["value"] = row["value"]
        new_row["error"] = row["error"]
        new_row[code] = row["value"]
        new_row[error_column] = row["error"]
        gt_df = pd.concat([gt_df, new_row.to_frame().T], ignore_index=True)
    print(f"已追加 {len(unmatched_df)} 条未匹配数据在末尾")

    # 保存比较结果
    toeval_df["material"] = toeval_df["original_material"]
    toeval_df.drop(columns=["original_material"], inplace=True)
    with pd.ExcelWriter(comparison_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        gt_df.to_excel(writer, sheet_name="summary", index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        toeval_df.to_excel(writer, sheet_name="4-match", index=False)
    print(f"处理完成，{code} 共有 {match_count}/{len(toeval_df)} 条匹配。")
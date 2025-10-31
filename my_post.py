import pandas as pd
import re


def clean_and_normalize(xlsx_path, comparison_xlsx, code, sheet_name=0):
    """Clean and normalize extracted data.
    
    Unified post-processing pipeline starting from raw DOI, material, and value (with unit) columns.
    
    Args:
        xlsx_path (str): Path to result Excel file (format: "FINAL_{CODE}_{dtime}.xlsx")
        comparison_xlsx (str): Path to comparison Excel file for validation
        code (str): Extraction method code identifier
        sheet_name (int or str): Sheet to process containing doi, material, value (with unit) columns
    """
    # Step 0: Basic data cleaning
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Normalize various Unicode dash characters to standard hyphen
    special_dashes = ["−", "–", "—", "‐", "‑", "⁃", "₋", "⁻"]
    for dash in special_dashes:
        df["material"] = df["material"].str.replace(dash, "-")
    
    df = df.drop_duplicates()
    df["doi"] = df["doi"].str.replace("_", "/")
    df["del"] = False

    # Step 1: Separate value and unit into distinct columns
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

    # Step 2: Apply filtering criteria to mark rows for deletion
    # Filter 1: Remove entries without eV or meV units
    df.loc[~df["unit"].str.contains("eV", case=False, na=False), "del"] = True
    
    # Filter 2: Remove specific material keywords (non-materials or ambiguous terms)
    keywords = ['brookite', 'Brookite', 'nitrogen', 'Nitrogen', 'oxygen', 'Oxygen', 
               'VB', 'VBM', 'TM', 'transition metal', 'VOC']
    df.loc[df["material"].isin(keywords), "del"] = True
    
    # Filter 3: Remove materials ending with charge indicators (+/-)
    df.loc[df["material"].str[-1].isin(["+", "-"]), "del"] = True
    
    # Filter 4: Remove short material names (1-2 chars) except allowed elements
    allowed_names = ["Ge", "Sn", "Si", "Se", "Te", "C", "S"]
    df.loc[(df['material'].str.len().isin([1, 2])) & (~df['material'].isin(allowed_names)), 'del'] = True
    
    # Filter 5: Remove entries without numeric values
    df.loc[~df["value"].apply(lambda x: bool(re.search(r"\d", str(x)))), "del"] = True

    # Reorder columns and save initial cleaning results
    new_order = ["doi", "material", "value", "unit", "del"]
    new_order += [col for col in df.columns if col not in new_order]
    df = df[new_order]
    with pd.ExcelWriter(xlsx_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name="2-clean", index=False)
    
    df_clean = df[~df["del"]].drop(columns=["del"])
    print(f"Initial filtering removed: {len(df[df['del']])}/{len(df)} entries")
    
    # Check if all rows were marked for deletion
    if df['del'].all():
        print("All rows marked for deletion. Outputting empty table and skipping further processing.")
        empty_df = pd.DataFrame(columns=df.columns)
        with pd.ExcelWriter(xlsx_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
            empty_df.to_excel(writer, sheet_name="3-normalize", index=False)
        with pd.ExcelWriter(comparison_xlsx, mode="a", if_sheet_exists="replace", engine="openpyxl") as writer:
            empty_df.to_excel(writer, sheet_name=f"{code}_clean", index=False)
        return

    # Step 3: Normalize value column
    def normalize_value(value):
        """Standardize value strings for consistent formatting.
        
        Performs the following normalizations:
        - Unifies various Unicode symbols to ASCII equivalents
        - Converts text representations to symbols (e.g., "approximately" -> "~")
        - Standardizes range notation (e.g., "1.5 to 2.3" -> "1.5-2.3")
        - Removes extraneous parentheses and periods
        - Removes all whitespace
        
        Args:
            value: Input value string
            
        Returns:
            str: Normalized value string
        """
        value = str(value)
        
        # Unify Unicode symbols to ASCII equivalents
        char_map = {
            "−": "-", "–": "-", "—": "-", "～": "~", "∼": "~", "≈": "~",
            "＜": "<", "＞": ">", "±": "±"
        }
        for old, new in char_map.items():
            value = value.replace(old, new)
            
        # Convert text comparisons to symbols
        text_map = {
            "below": "<", "less than": "<", "lower than": "<", "under": "<",
            "above": ">", "greater than": ">", "over": ">",
            "about": "~", "approximately": "~", "ca.": "~", "circa": "~"
        }
        for text, symbol in text_map.items():
            value = re.sub(rf"\b{re.escape(text)}\b", symbol, value, flags=re.IGNORECASE)
        
        # Standardize range notation
        value = re.sub(r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)", r"\1-\2", value, flags=re.IGNORECASE)
        value = re.sub(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", r"\1-\2", value)
        
        # Remove extraneous characters
        value = value.strip("()")
        value = value.strip(".")
        value = value.replace(" ", "")
        
        return value

    df_clean["value"] = df_clean["value"].apply(normalize_value)

    # Step 4: Handle multi-value entries by splitting into separate rows
    new_rows = []
    for _, row in df_clean.iterrows():
        value = row["value"]
        
        if isinstance(value, str):
            # Replace "and" with comma for consistent splitting
            value = value.replace("and", ",")
            split_values = value.split(",")
            cleaned_values = [v.strip() for v in split_values if v.strip()]
            
            # Check if all parts are numeric (allowing symbols and ranges)
            all_numeric = True
            for v in cleaned_values:
                if not re.match(r"^[<>~±-]?\d*\.?\d+([eE][-+]?\d+)?$", v) and not "-" in v:
                    all_numeric = False
                    break
            
            # Split into multiple rows if all values are numeric
            if all_numeric and len(cleaned_values) > 1:
                for v in cleaned_values:
                    new_row = row.copy()
                    new_row["value"] = v
                    new_rows.append(new_row)
                continue
        
        new_rows.append(row)
    
    df_clean = pd.DataFrame(new_rows).reset_index(drop=True)

    # Step 5: Convert units (meV to eV, kV to V, etc.) while preserving precision
    def convert_unit(value, unit):
        """Convert values with metric prefixes to base units.
        
        Handles conversion from meV/keV to eV while preserving significant figures
        and non-numeric components of the value string.
        
        Args:
            value: Value string (may contain non-numeric characters)
            unit (str): Unit string (e.g., "meV", "keV", "eV")
            
        Returns:
            tuple: (converted_value, converted_unit)
        """
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

    # Step 6: Remove rows with empty values
    original_count = len(df_clean)
    df_clean.dropna(subset=['value'], inplace=True)
    deleted_rows = original_count - len(df_clean)
    print(f"Removed rows with empty values: {deleted_rows}")
    
    # Step 7: Remove duplicate entries (same DOI, material, and value)
    original_count = len(df_clean)
    df_clean.drop_duplicates(subset=["doi", "material", "value"], keep='first', inplace=True)
    deleted_rows = original_count - len(df_clean)
    print(f"Removed duplicate entries: {deleted_rows}")
    
    # Reorder columns for final output
    new_order = ["doi", "material", "value"]
    new_order += [col for col in df_clean.columns if col not in new_order]
    df_clean = df_clean[new_order]
    
    # Save normalized results
    with pd.ExcelWriter(xlsx_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        df_clean.to_excel(writer, sheet_name="3-normalize", index=False)
    with pd.ExcelWriter(comparison_xlsx, mode="a", if_sheet_exists="replace", engine="openpyxl") as writer:
        df_clean.to_excel(writer, sheet_name=f"{code}_clean", index=False)



def compare_with_index(xlsx_path, comparison_xlsx, code):
    """Compare extracted results with ground truth data.
    
    Validates extraction results against a reference dataset, matching entries
    by DOI, material name, and value. Handles alternative material names and
    appends unmatched entries to the ground truth file.
    
    Args:
        xlsx_path (str): Path to result Excel file (format: "FINAL_{CODE}_{dtime}.xlsx")
        comparison_xlsx (str): Path to ground truth Excel file
        code (str): Extraction method code identifier
    """
    error_column = code + "_error"
    gt_df = pd.read_excel(comparison_xlsx, sheet_name="summary")
    toeval_df = pd.read_excel(xlsx_path, sheet_name=-1)
    
    # Parse DOI column to extract index and clean DOI
    split_result = toeval_df["doi"].str.split('/', n=1, expand=True)
    toeval_df["index"] = split_result[0]
    toeval_df["doi"] = split_result[1]
    
    toeval_df["index"] = pd.to_numeric(toeval_df["index"], errors='coerce')
    gt_df["index"] = pd.to_numeric(gt_df["index"], errors='coerce')
    
    toeval_df["match"] = ""
    gt_df["value"] = gt_df["value"].astype(str)
    toeval_df["value"] = toeval_df["value"].astype(str)
    
    # Preserve original material names and normalize for comparison
    toeval_df["original_material"] = toeval_df["material"].copy()
    toeval_df.loc[:, "material"] = toeval_df["material"].apply(lambda x: x.lower() if isinstance(x, str) else x)
    
    # Primary matching: Compare with ground truth using main material column
    match_count = 0
    for idx, gt_row in gt_df.iterrows():
        comp_material = gt_row['material']
        # Handle semicolon-separated material synonyms
        comp_materials = [m.strip().lower() if isinstance(m.strip(), str) else m.strip()
                         for m in comp_material.split(';')] if isinstance(comp_material, str) else [comp_material]
        
        # Compare values (numeric or string)
        gt_value = gt_row["value"]
        try:
            gt_value_num = float(gt_value)
            value_comparison = toeval_df["value"].apply(lambda x: float(x) if isinstance(x, str) and (x.replace('-','',1).replace('.','',1).isdigit()) else x) == gt_value_num
        except ValueError:
            value_comparison = toeval_df["value"] == gt_value
            
        match = toeval_df[
            (toeval_df["index"] == gt_row["index"]) &
            (toeval_df["doi"] == gt_row["doi"]) &
            (toeval_df["material"].isin(comp_materials)) &
            value_comparison
        ]
        if not match.empty:
            gt_df.at[idx, code] = match.iloc[0]["value"]
            match_count += 1
            toeval_df.loc[match.index, "match"] = 1

    # Secondary matching: If ground truth has alternative material names (other_mat column)
    if "other_mat" in gt_df.columns:
        unmatched_df = toeval_df[toeval_df["match"] != 1]
        
        for idx, gt_row in gt_df.iterrows():
            if pd.notna(gt_row["other_mat"]):
                comp_other_mat = gt_row['other_mat']
                comp_other_mats = [m.strip().lower() if isinstance(m.strip(), str) else m.strip()
                                for m in comp_other_mat.split(';')] if isinstance(comp_other_mat, str) else [comp_other_mat]
                
                gt_value = gt_row["value"]
                try:
                    gt_value_num = float(gt_value)
                    value_comparison = unmatched_df["value"].apply(lambda x: float(x) if isinstance(x, str) and (x.replace('-','',1).replace('.','',1).isdigit()) else x) == gt_value_num
                except ValueError:
                    value_comparison = unmatched_df["value"] == gt_value
                    
                match = unmatched_df[
                    (unmatched_df["index"] == gt_row["index"]) &
                    (unmatched_df["doi"] == gt_row["doi"]) &
                    (unmatched_df["material"].isin(comp_other_mats)) &
                    value_comparison
                ]
                if not match.empty:
                    gt_df.at[idx, code] = match.iloc[0]["value"]
                    match_count += 1
                    toeval_df.loc[match.index, "match"] = 2
    
    # Append unmatched entries to ground truth file
    unmatched_df = toeval_df[(toeval_df["match"] != 1) & (toeval_df["match"] != 2)]
    
    if "other_mat" not in gt_df.columns:
        material_idx = gt_df.columns.get_loc("material")
        gt_df.insert(material_idx, "other_mat", "")
    
    for _, row in unmatched_df.iterrows():
        new_row = pd.Series(index=gt_df.columns)
        new_row["index"] = row["index"]
        new_row["doi"] = row["doi"]
        new_row["other_mat"] = row["original_material"]
        new_row["value"] = row["value"]
        new_row[code] = row["value"]
        gt_df = pd.concat([gt_df, new_row.to_frame().T], ignore_index=True)
    
    print(f"Appended {len(unmatched_df)} unmatched entries to ground truth file")

    # Sort results for readability
    if 'index' in toeval_df.columns:
        cols = ['index'] + [col for col in toeval_df.columns if col != 'index']
        toeval_df = toeval_df[cols]
    
    sort_columns = ['index', 'material', 'other_mat']
    gt_df = gt_df.sort_values(by=[col for col in sort_columns if col in gt_df.columns])
    
    # Restore original material names and save results
    toeval_df["material"] = toeval_df["original_material"]
    toeval_df.drop(columns=["original_material"], inplace=True)
    
    with pd.ExcelWriter(comparison_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        gt_df.to_excel(writer, sheet_name="summary", index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        toeval_df.to_excel(writer, sheet_name="4-match", index=False)
    
    print(f"Processing complete. {code}: {match_count}/{len(toeval_df)} entries matched.")



def parse_markdown_table(output, third="unit"):
    """Parse a markdown-formatted table into a pandas DataFrame.
    
    Extracts tabular data from markdown table strings, handling malformed rows
    and setting default values for missing columns.
    
    Args:
        output (str): Markdown-formatted table string
        third (str): Name of the third column (default: "unit")
        
    Returns:
        pd.DataFrame: Parsed table with columns [material, value, third],
                      or None if parsing fails
    """
    default_value = "eV" if third == "unit" else ""
    
    try:
        # Extract table content between first and last pipe characters
        start_index = output.find('|')
        end_index = output.rfind('|') + 1
        
        if start_index == -1 or end_index == -1:
            return None
        
        cleaned_output = output[start_index:end_index].strip()
        lines = cleaned_output.split('\n')
        
        # Parse data rows (skip header and separator rows)
        data = []
        for line in lines[2:]:
            stripped_line = line.strip('| ')
            parts = [p.strip() for p in stripped_line.split('|')]
            
            if len(parts) == 3:
                data.append(parts)
            elif len(parts) < 3:
                # Pad with None to ensure 3 columns
                while len(parts) < 3:
                    parts.append(None)
                data.append(parts)
            elif len(parts) > 3:
                # Handle rows with extra pipes (e.g., in sentence content)
                material = parts[0].strip()
                value = parts[1].strip()
                sentence = '|'.join(parts[2:]).strip()
                data.append([material, value, sentence])
            else:
                continue

        if not data:
            return None
            
        table = pd.DataFrame(data, columns=["material", "value", third])
        table = table.map(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Set default unit if third column is empty for all rows
        if third == "unit" and table[third].eq("").all():
            table[third] = default_value
            
        return table
        
    except Exception as e:
        print(f"Error parsing markdown table:\n{output}\nError: {str(e)}")
        return None


def clean_illegal_chars(text):
    """Remove illegal control characters from text.
    
    Removes non-printable control characters while preserving tabs, newlines,
    and carriage returns. Useful for sanitizing text before Excel export.
    
    Args:
        text: Input text (any type)
        
    Returns:
        Cleaned text string, or original value if not a string
    """
    if isinstance(text, str):
        # Remove control characters (preserve \t, \n, \r)
        return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]", "", text)
    else:
        return text
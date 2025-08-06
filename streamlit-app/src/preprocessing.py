def prepare_data(df, mode="brand", id="B1"):
    df = df.copy()
    if mode == "brand":
        qty_cols = [col for col in df.columns if col.startswith(f"QTY_{id}_")]
        promo_cols = [col for col in df.columns if col.startswith(f"PROMO_{id}_")]
        df[f"QTY_{mode.upper()}_{id}"] = df[qty_cols].sum(axis=1)
        df[f"PROMO_{mode.upper()}_{id}"] = df[promo_cols].max(axis=1)
        result = df[[f"QTY_{mode.upper()}_{id}", f"PROMO_{mode.upper()}_{id}"]].copy()
    elif mode == "sku":
        result = df[[f"QTY_{id}", f"PROMO_{id}"]].copy()
        result = result.rename(columns={
            f"QTY_{id}": f"QTY_{mode.upper()}_{id}",
            f"PROMO_{id}": f"PROMO_{mode.upper()}_{id}"
        })
    result.index = df.index
    return result

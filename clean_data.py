import pandas as pd
import os

# Input: the Kaggle file "Zameen.com Property Data Pakistan 2023" saved as data/raw/archive/zameen.csv
df = pd.read_csv("data/raw/archive/zameen.csv", sep="|")

# 1. Karachi only
df = df[df["city"] == "Karachi"].copy()
print("Karachi rows (start):", len(df))

# 2. Remove odd values and duplicates
df = df[(df["price"] >= 1_000_000) & (df["price"] <= 1_000_000_000)]
df = df[(df["size"] >= 100) & (df["size"] <= 50_000)]
df = df.drop_duplicates()
print("After cleaning:", len(df))

# 3. Split the location text
df["area"] = df["location"].str.split(",").str[0].str.strip()


def get_society(loc):
    if "DHA" in loc:
        return "DHA"
    if "Bahria" in loc:
        return "Bahria Town"
    if "Askari" in loc:
        return "Askari 5"
    return loc.split(",")[-1].strip()


df["society"] = df["location"].apply(get_society)


# 4. Property type is inferred (the dataset has no type column)
def get_type(row):
    if row["bedrooms"] == 0:
        return "Plot"
    kw = ["Heights", "Apartment", "Tower", "Residency", "Flat"]
    if any(k in row["location"] for k in kw):
        return "Flat"
    return "House"


df["type"] = df.apply(get_type, axis=1)

# 5. Size: the dataset is in square feet, Karachi uses square yards (1 sq yd = 9 sq ft)
df["size_sqft"] = df["size"].round(0).astype(int)
df["size_sqyd"] = (df["size"] / 9).round(0).astype(int)


# 6. Price in lakh / crore
def fmt_price(p):
    if p >= 10_000_000:
        return f"{p / 10_000_000:g} crore"
    return f"{p / 100_000:g} lakh"


df["price_text"] = df["price"].apply(fmt_price)


# 7. One short text per listing (this is what the search reads)
def describe(r):
    place = r["area"]
    if r["society"] not in place and r["society"] != "Karachi":
        place += f", {r['society']}"

    if r["type"] == "Plot":
        return (f"Plot in {place}, Karachi. "
                f"Size {r['size_sqyd']} sq yd ({r['size_sqft']} sq ft). "
                f"Price {r['price_text']}.")
    return (f"{r['type']} in {place}, Karachi. "
            f"Size {r['size_sqyd']} sq yd ({r['size_sqft']} sq ft), "
            f"{r['bedrooms']} bedrooms, {r['baths']} bathrooms. "
            f"Price {r['price_text']}.")


df["description"] = df.apply(describe, axis=1)

# 8. IDs and save
df = df.reset_index(drop=True)
df.insert(0, "id", ["P" + str(i + 1).zfill(4) for i in range(len(df))])

os.makedirs("data/clean", exist_ok=True)
cols = ["id", "society", "area", "type", "size_sqyd", "size_sqft",
        "bedrooms", "baths", "price", "price_text", "description"]
df[cols].to_csv("data/clean/properties.csv", index=False)

print("\nSociety-wise:")
print(df["society"].value_counts().head(8))
print("\nType-wise:")
print(df["type"].value_counts())
print("\n5 sample descriptions:")
for d in df["description"].sample(5, random_state=1):
    print("-", d)

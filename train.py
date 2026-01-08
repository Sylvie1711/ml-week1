import pandas as pd

def main():
    df = pd.read_csv("data/data.csv")

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nTarget stats:")
    print(df["median_house_value"].describe())

if __name__ == "__main__":
    main()

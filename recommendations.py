import pandas as pd

# Read the CSV file into a DataFrame
file_path = 'userproduct.csv'
df = pd.read_csv(file_path, index_col=0)

# Convert 'Y' and 'N' to 1 and 0
df = df.replace({'Y': 1, 'N': 0})

# Calculate the co-existence matrix
coexistence_matrix = df.T.dot(df)

# Display the co-existence matrix
print("Co-Existence Matrix:")
print(coexistence_matrix)

def get_recommendations(coexistence_matrix, target_product, num_recommendations=3):
    if target_product not in coexistence_matrix.index:
        raise KeyError(f"Product {target_product} not found in the co-existence matrix.")
    
    # Sort products by co-existence with the target product
    recommended_products = coexistence_matrix[target_product].sort_values(ascending=False)
    
    # Exclude the target product itself
    recommended_products = recommended_products[recommended_products.index != target_product]
    
    # Get the top N recommendations
    recommendations = recommended_products.head(num_recommendations)
    
    return recommendations

# Example: Get recommendations for a user who just bought Mobile Model 1
target_product = 'Mobile Model 1'
recommendations = get_recommendations(coexistence_matrix, target_product)

# Display the recommendations
print(f"Recommendations for a user who just bought {target_product}:")
print(recommendations)

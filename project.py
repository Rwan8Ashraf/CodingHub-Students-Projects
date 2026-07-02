import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Task 2:
# Read the dataset supermarket_sales.csv.
df = pd.read_csv('supermarket.csv')

# Task 3:
# Display the first 10 rows.
print(df.head(10))

print("--------------------------------------------------")
# Task 4:
# Display the last 10 rows.
print(df.tail(10))

print("--------------------------------------------------")
# Task 5:
# Print the dataset information.
print(df.info())

print("--------------------------------------------------")
# Task 6:
# Print the statistical summary of all numerical columns.
print(df.describe())

print("--------------------------------------------------")
# Task 7:
# Print:
# 
#  Shape
# 
#  Columns
# 
#  Data Types

print("shape: ", df.shape)
print("columns: ", df.columns)
print("data types: ", df.dtypes)

print("--------------------------------------------------")

# Part 2: Access Data
# Task 8:
# Display only the Product_Line column.
print(df['Product_Line'])

print("--------------------------------------------------")

# Task 9:
# Display the following columns:
# 
#  Product_Line
# 
#  Quantity
# 
#  Net_Total
print(df[['Product_Line', 'Quantity', 'Net_Total']])

print("--------------------------------------------------")

# [1, 3, 45, 6]
# Task 10:
# Print sales records from row 40 to row 70.
print(df.iloc[40:71])

print("--------------------------------------------------")
# Part 3: Filtering
# Task 11:
# Display sales where Net_Total is greater than 2000.
print(df[df['Net_Total'] > 2000])

print("--------------------------------------------------")
# Task 12:
# Display sales where Quantity is greater than 5.
print(df[df['Quantity'] > 5])

print("--------------------------------------------------")

# Task 13:
# Display all sales from Branch A.
print(df[df['Branch'] == 'A'])

print("--------------------------------------------------")
# Task 14:
# Display all sales paid using Credit Card.
print(df[df['Payment'] == 'Credit card'])

print("--------------------------------------------------")
# Task 15:
# Display sales where the customer is a Member.
print(df[df['Customer_Type'] == 'Member'])

print("--------------------------------------------------")
# Task 16:
# Display sales where the Rating is greater than 9.
print(df[df['Rating'] > 9])

print("--------------------------------------------------")
# Task 17:
# Display sales where Quantity is between 3 and 7.
print(df[(df['Quantity'] >= 3) & (df['Quantity'] <= 7)])

print("--------------------------------------------------")

#df["test"] = 5
#df["dis"] = df["Discount"] / df["Total"] * 100
# Part 4: Add New Columns
# Task 18:
# Create a new column called:
# Discount_Percentage
# Use the following formula:
# Discount_Percentage = (Discount / Total) * 100
df["Discount_Percentage"] = (df["Discount"] / df["Total"]) * 100

print(df.head())

print("--------------------------------------------------")
# Part 5: NumPy
# Task 19:
# Convert Net_Total into a NumPy array.
sales = df["Net_Total"].to_numpy()
# Find:
# 
#  Maximum
# 
#  Minimum
# 
#  Mean
# 
#  Sum
print("Maximum Net_Total: ", np.max(sales))
print("Minimum Net_Total: ", np.min(sales))
print("Mean Net_Total: ", np.mean(sales))
print("Sum Net_Total: ", np.sum(sales))

print("--------------------------------------------------")
# Part 6: GroupBy
# Task 20:
# Calculate the average Net_Total for each Branch.
print(df["Net_Total"].groupby(df["Branch"]).mean())

print("--------------------------------------------------")
# Task 21:
# Calculate the average Rating for each Customer_Type.
print(df["Rating"].groupby(df["Customer_Type"]).mean())

print("--------------------------------------------------")
# Task 22:
# Find the highest Net_Total for each Product_Line.
print(df["Net_Total"].groupby(df["Product_Line"]).max())

print("--------------------------------------------------")
# Task 23:
# Find the minimum Unit_Price for each Product_Line.
print(df["Unit_Price"].groupby(df["Product_Line"]).min())

print("--------------------------------------------------")
# Task 24:
# Count the number of sales for each Payment method.
print(df["Invoice_ID"].groupby(df["Payment"]).count())

print("--------------------------------------------------")

print(df["Payment"].value_counts())

print("--------------------------------------------------")
# Part 7: Visualization (Matplotlib)
# Task 25:
# Draw a Bar Chart showing the average Net_Total for each Branch.

averge_sales= df["Net_Total"].groupby(df["Branch"]).mean()
plt.bar(averge_sales.index, averge_sales.values)
plt.title("Average Net_Total for each Branch")
plt.xlabel("Branch")
plt.ylabel("Average Net_Total")
plt.show()

# Task 26:
# Draw a Histogram of Net_Total.
plt.hist(df["Net_Total"], bins= 20)
plt.title("Histogram of Net_Total")
plt.xlabel("Net_Total")
plt.ylabel("Frequency")
plt.show()

# Task 27:
# Draw a Line Plot showing Invoice_ID vs Net_Total.
plt.plot(df["Invoice_ID"], df["Net_Total"])
plt.title("Invoice_ID vs Net_Total")
plt.xlabel("Invoice_ID")    
plt.ylabel("Net_Total")
plt.show()
# Part 8: Visualization (Seaborn)
# Task 28:
# Draw a Scatter Plot between Quantity and Net_Total.
sns.scatterplot(data= df, x="Quantity", y="Net_Total")
plt.title("Scatter Plot between Quantity and Net_Total")
plt.show()

# Task 29:
# Draw a Bar Plot showing the average Net_Total for each Product_Line.
sns.barplot(data= df, x="Product_Line", y="Net_Total", ci=None)
plt.title("Average Net_Total for each Product_Line")
plt.show()

# Task 30:
# Draw a Histogram of Net_Total with KDE.
sns.histplot(data= df, x="Net_Total", kde=True, bins= 20)
plt.title("Histogram of Net_Total with KDE")
plt.show()

# Part 9: Save Figure
# Task 31:
# Save the Histogram as:
# sales_histogram.png
# using:
# dpi = 300
plt.savefig("sales_histogram.png", dpi=300)
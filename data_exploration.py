from ucimlrepo import fetch_ucirepo
parkinsons = fetch_ucirepo(id=174)
X = parkinsons.data.features
y = parkinsons.data.targets

print("X.head")
print(X.head())

print("X.info")
print(X.info())

print("y.head")
print(y.head())

print("y.value")
print(y.value_counts())


# Print metadata
#print("UCI ID:", parkinsons
#
# .m]
#metadata.uci_id)
#print("Number of Instances:", parkinsons.metadata.num_instances)
#print("Summary:", parkinsons.metadata.additional_info.summary)

# variable information
#print(parkinsons.variables)





#logic = LinearRegression()
#logic.fit(X, y)
#print("Model trained successfully.")
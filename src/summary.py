import pandas as pd
import matplotlib.pyplot as plt

# Top 50 scatter with color by off-target hits
df = pd.read_csv("cd36_sgrnas.csv")
print(df["Efficiency"].describe())
top50 = df.head(50)

plt.figure(figsize=(8, 5))
top50["Efficiency"].hist(bins=50)
plt.title("Distribution of Predicted sgRNA Efficiencies")
plt.xlabel("Efficiency Score")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("efficiency_histogram_cd36.png")
plt.show()



# df = pd.read_csv("tp53_sgrnas.csv")
# df = df.sort_values("Efficiency", ascending=False).head(50)

# plt.figure(figsize=(8, 5))
# sc = plt.scatter(range(len(df)), df["Efficiency"], c=df["OffTarget_Hits"], cmap="coolwarm", s=80, edgecolor='k')
# plt.colorbar(sc, label="Off-Target Hits")
# plt.title("Top 50 sgRNAs - Efficiency vs Off-Target Risk")
# plt.xlabel("sgRNA Rank")
# plt.ylabel("Efficiency Score")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("tp53_scatter_efficiency_vs_offtargets.png")
# plt.show()


# Efficiency distributions for multiple gene - side by side plots
# genes = ["tp53", "cd36"]  # here to add more gene CSVs if needed
# dataframes = {}

# for gene in genes:
#     df = pd.read_csv(f"{gene}_sgrnas.csv")
#     df = df.sort_values("Efficiency", ascending=False)
#     dataframes[gene] = df.head(50)

# # Plotting
# fig, axs = plt.subplots(1, len(genes), figsize=(6 * len(genes), 5))

# for i, gene in enumerate(genes):
#     ax = axs[i]
#     df = dataframes[gene]
#     df["Efficiency"].hist(bins=20, ax=ax)
#     ax.set_title(f"{gene.upper()} - Top 50 sgRNAs")
#     ax.set_xlabel("Efficiency Score")
#     ax.set_ylabel("Frequency")

# plt.tight_layout()
# plt.savefig("comparison_histograms_top50.png")
# plt.show()
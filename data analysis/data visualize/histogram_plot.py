import matplotlib.pyplot as plt

def histogram_plot (data, label="Histogram plot"):
  plt.figure(figsize=(6, 4))
  plt.hist(data, bins=4, color='skyblue', edgecolor='black')
  plt.title(f"{label} Distribution")
  plt.xlabel(label)
  plt.ylabel("Frequency")
  plt.grid(axis='y', alpha=0.75)
  plt.show()
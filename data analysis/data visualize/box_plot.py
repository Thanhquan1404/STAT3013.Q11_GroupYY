import matplotlib.pyplot as plt

def box_plot(data, label="Box plot"):
  plt.figure(figsize=(6, 4))
  plt.boxplot(data, vert=False, patch_artist=True,
              boxprops=dict(facecolor='lightgreen', color='black'),
              medianprops=dict(color='red'))
  plt.title(f"Boxplot of {label}")
  plt.xlabel(label)
  plt.show()
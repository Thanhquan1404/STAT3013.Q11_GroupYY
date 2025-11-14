import matplotlib.pyplot as plt
import seaborn as sns


def density_plot(data, feature_name="feature_name"):
  fig = plt.figure(figsize=(14, 10))
  grid = plt.GridSpec(3, 3, wspace=0.3, hspace=0.4)
  ax5 = fig.add_subplot(grid[1, 2])
  sns.kdeplot(data, fill=True, ax=ax5)
  ax5.set_title(f"Density Plot (KDE): {feature_name}")
  ax5.set_xlabel(feature_name)
  ax5.set_ylabel("Density")
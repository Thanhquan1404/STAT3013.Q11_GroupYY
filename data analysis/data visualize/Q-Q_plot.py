import matplotlib.pyplot as plt
import scipy.stats as stats

def QQ_plot (data, label="Q-Q plot"):
  stats.probplot(data, dist="norm", plot=plt)
  plt.title(label)
  plt.show()
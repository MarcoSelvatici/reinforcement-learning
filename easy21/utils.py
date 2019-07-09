import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as axes3d
import pandas as pd

def plot(Q):
  x_range = list(range(1,10))
  y_range = list(range(1,22))
  v_star = list()
  for x in x_range:
    for y in y_range:
      v_star.append( [x, y, max([Q[x, y, a] for a in [0, 1]])] )
  
  df = pd.DataFrame(v_star, columns=['dealer', 'player', 'value'])

  # Make the plot.
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot_trisurf(df['player'], df['dealer'], df['value'], cmap=plt.cm.viridis, \
                  linewidth=0.2)
  plt.show()

  # to Add a color bar which maps values to colors.
  surf=ax.plot_trisurf(df['player'], df['dealer'], df['value'], cmap=plt.cm.viridis, \
                       linewidth=0.2)
  fig.colorbar( surf, shrink=0.5, aspect=5)
  plt.show()

  # Rotate it.
  ax.view_init(30, 45)
  plt.show()

  # Other palette.
  ax.plot_trisurf(df['player'], df['dealer'], df['value'], cmap=plt.cm.jet, \
                  linewidth=0.01)
  plt.show()
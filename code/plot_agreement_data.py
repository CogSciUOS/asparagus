from plotnine import *
import pandas as pd
import glob
import os

agreement = pd.concat(map(pd.read_csv, glob.glob(os.path.join(
    '../annotations/evaluation_agreement_2', "agreement*.csv"))))

# print(agreement.head(30))


p = (ggplot(agreement, aes(x='evaluation_measure', y='score', group='annotators', color='feature'))
     + geom_line(aes(group='feature'), alpha=0.8)
     + geom_point(size=3, alpha=0.8)
     + facet_grid('.~annotators')
     + theme_bw())

# print(p)


p2 = (ggplot(agreement, aes(x='feature', y='score', group='feature', color='feature'))
      + geom_boxplot(aes(group='feature'))
      + facet_grid('.~evaluation_measure')
      + theme_bw())


# print(p2)


p3 = (ggplot(agreement, aes(x='evaluation_measure', y='score', group='evaluation_measure', color='evaluation_measure'))
      + geom_boxplot(aes(group='evaluation_measure'))
      + facet_grid('.~feature')
      + theme_bw()
      )


# print(p3)


kappa = agreement[agreement["evaluation_measure"] == "kappa"]


p4 = (ggplot(kappa, aes(x='feature', y='score', color='feature'))
      + geom_boxplot(aes(group='feature'))
      + theme_bw()
      )


print(p4)

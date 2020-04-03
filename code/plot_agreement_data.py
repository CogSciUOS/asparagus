from plotnine import *
import pandas as pd
import glob
import os

agreement = pd.concat(map(pd.read_csv, glob.glob(os.path.join(
    '../annotations/evaluation_agreement_2', "agreement*.csv"))))


p1 = (ggplot(agreement, aes(x='evaluation_measure', y='score', group='annotators', color='feature'))
      + geom_line(aes(group='feature'), alpha=0.7)
      + geom_point(size=3, alpha=0.7)
      + facet_grid('.~annotators')
      + ylim(0, 1.0)
      + xlab('evaluation measure')
      + labs(title='Annotator pairs vs. different measures')
      + theme_bw()
      + theme(legend_position='bottom', legend_box_margin=30, legend_direction='horizontal',
              legend_title=element_blank(), figure_size=(10, 6))
      + guides(color=guide_legend(nrow=1))
      )
# print(p1)
#
features_short = agreement.replace({'has_blume': 'blume', 'has_rost_body': 'r body',
                                    'has_rost_head': 'r head', 'is_bended': 'bended', 'is_hollow': 'hollow', 'is_violet': 'violet'})

p2 = (ggplot(features_short, aes(x='feature', y='score', group='feature', color='feature'))
      + geom_boxplot(aes(group='feature'))
      + facet_grid('.~evaluation_measure')
      + ylim(0, 1.0)
      + xlab('features')
      + labs(title='Comparing the different measures')
      + theme_bw()
      + theme(legend_position='none', figure_size=(12, 6))
      )
# print(p2)


p3 = (ggplot(features_short, aes(x='evaluation_measure', y='score', group='evaluation_measure', color='evaluation_measure'))
      + geom_boxplot(aes(group='evaluation_measure'))
      + facet_grid('.~feature')
      + ylim(0, 1.0)
      + xlab('features')
      + labs(title='Comparing the different features')
      + theme_bw()
      + theme(legend_position='none', figure_size=(12, 6))
      )
# print(p3)


# Filter out the kappa score
kappa = features_short[features_short["evaluation_measure"] == "kappa"]


p4 = (ggplot(kappa, aes(x='feature', y='score', color='feature'))
      + geom_boxplot(aes(group='feature'))
      + ylim(0, 1.0)
      + xlab('features')
      + labs(title='Aggregated kappa score over all annotator pairs')
      + theme_bw()
      + theme(legend_position='none', figure_size=(5, 6))
      )
# print(p4)


# save plots in a pdf
save_as_pdf_pages([p4, p1, p2, p3], filename="agreement_plots.pdf",
                  path="../annotations/evaluation_agreement_2")

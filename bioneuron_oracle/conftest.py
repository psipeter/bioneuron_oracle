from nengo.conftest import *  # noqa: F401, F403

import seaborn as sns
palette = sns.hls_palette(9, h=.6, l=.3, s=.9)
# palette = sns.palplot(sns.color_palette("bright", 9))
colors = ["blue", "red", "green", "yellow", "purple", "orange", "cyan", "hot pink", "tan"]
# colors = ["blue", "red", "green", "yellow", "purple", "orange", "cyan", "hot pink", "tan", "forest green", "dark blue", "gray", "brown", "navy blue", "light pink", "dark red", "gold", "black"]
palette = sns.xkcd_palette(colors)
sns.set(context='poster', palette=palette, style='whitegrid')

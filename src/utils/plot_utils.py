import plotnine as gg


# just overwriting gg.themes.theme_538 with some custom settings
class theme_538(gg.themes.theme_gray):
    def __init__(self, base_size=11, base_family='DejaVu Sans'):
        gg.themes.theme_gray.__init__(self, base_size, base_family)
        bgcolor = '#FFFFFF'
        self.add_theme(
            gg.themes.theme(
                axis_ticks=gg.element_blank(),
                title=gg.element_text(color='#3C3C3C'),
                legend_background=gg.element_rect(fill="white", colour="black", size=0.5),
                legend_key=gg.element_rect(fill='#E0E0E0'),
                panel_background=gg.element_rect(fill=bgcolor),
                panel_border=gg.element_blank(),
                panel_grid_major=gg.element_line(
                    color='#D5D5D5', linetype='solid', size=1),
                panel_grid_minor=gg.element_blank(),
                plot_background=gg.element_rect(
                    fill=bgcolor, color=bgcolor, size=1),
                strip_background=gg.element_rect(size=0)),
            inplace=True)

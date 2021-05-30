import plotly.graph_objects as go
from Lib.utils.Logger.Logger import Logger


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This class is thought to compare different Logger instance
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Logger_cmp(Logger):
    """
    This class is an extention of Logger, it is used to compare logger. Basically is composed of a list of Logger and has
    some methods to plot each Logger against another
    """
    def __init__(self, mode, shades, logger_list, name_list, title):
        """

        :param mode: not implemented yet
        :param shades: number of shades so basically number of signal for each Logger
        :param logger_list: logger list
        :param name_list: name of each logger
        :param title: general title
        The title of the graph is composed: title + name_list[0] + VS + name_list[1] + VS + ... + name_list[N]
        """
        self.mode = mode
        self.shades = shades
        self.shade_name = []
        self.color_shade = self.get_shades()
        self.base_color = self.get_base_color()
        self.logger_list = logger_list
        self.name_list = name_list
        self.title = title
        self.len = len(logger_list)

    def create_figure(self):
        """
        This method creates the figure of the loss function and the accuracy
        :return:
        """
        if self.mode == "trainall":
            self.fig_G_loss = self.generate_plot(self.G_loss_df)
            self.fig_G_loss = self.plot_layout(self.fig_G_loss, "Generator Losses")
            self.fig_D_S_loss = self.generate_plot(self.D_S_loss_df)
            self.fig_D_S_loss = self.plot_layout(self.fig_D_S_loss, "SAR Discriminator Losses")
            self.fig_D_O_loss = self.generate_plot(self.D_O_loss_df)
            self.fig_D_O_loss = self.plot_layout(self.fig_D_O_loss, "Optical Discriminator Losses")

        self.fig_SN_loss = self.generate_plot('SN_loss_df')
        title = self.get_title(self.title)
        self.fig_SN_loss = self.plot_layout(self.fig_SN_loss, title)

        self.fig_SN_acc = self.generate_plot('SN_acc_df')
        title = self.get_title(self.title)
        self.fig_SN_acc = self.plot_layout(self.fig_SN_acc, title, "Epochs", "OA [%]")

    def generate_plot(self, var):
        """
        This function iterate over each signal of each logger and add them to the plot assigning always different colour
        :param var: name of the variable in logger which contains the x axis
        :return:
        """
        fig = go.Figure()
        for j, logger in enumerate(self.logger_list):
            df = getattr(logger, var)
            if 'acc' in var:
                df_x = getattr(logger, 'acc_step_df')
            else:
                df_x = getattr(logger, 'loss_step_df')
            self.epoch = logger.epoch
            df_x = self.norm_step(df_x)
            #shades = self.color_shade[self.shade_name[j]]
            shades = self.base_color
            for i, col in enumerate(df.columns):
                color = color=shades[j]
                if i==0:
                    #TODO: which metrics plot, *100
                    fig.add_trace(go.Scatter(x=df_x['step'], y=df[col]*100, mode='lines+markers', line=dict(color=color, width=4),
                                             marker=dict(color=color, size=10), name=col, ))
        return fig

    def get_title(self, title):
        """
        Create the title from the name list
        :param title:
        :return:
        """
        print(title)
        '''
        if title != "":
            for i in range(self.len - 1):
                title = title + self.name_list[i] + '  <b>VS</b>  '
            title = title + self.name_list[-1]'''
        return title

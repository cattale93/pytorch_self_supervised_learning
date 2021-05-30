import pandas as pd
import pickle as pkl
from Lib.utils.generic.generic_utils import color_hex_list_generator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Lib.utils.generic.generic_utils import moving_average
import os


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This class is thought to log data during training, then reload the results and plot them
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Logger:
    """
    This class is thought to store and plot data.
    The storing is made during training execution, the plotting is performed after
    """
    def __init__(self, mode='', shades=5):
        self.mode = mode
        self.epoch = None
        self.shades = shades
        self.shade_name = []
        self.color_shade = self.get_shades()
        self.base_color = self.get_base_color()
        self.fig_G_loss = None
        self.fig_D_S_loss = None
        self.fig_D_O_loss = None
        self.fig_GAN = None
        self.fig_SN_loss = None
        self.fig_SN_acc = None
        self.fig_SN = None
        self.loss_step_df = pd.DataFrame()
        self.acc_step_df = pd.DataFrame()
        self.SN_loss_df = pd.DataFrame()
        self.SN_acc_df = pd.DataFrame()
        if self.mode == "train":
            self.G_loss_df = pd.DataFrame()
            self.D_S_loss_df = pd.DataFrame()
            self.D_O_loss_df = pd.DataFrame()

    # All these function append a dictionary which is a sample of each of these acquired variable
    def append_G(self, new_sample):
        self.G_loss_df = self.G_loss_df.append(new_sample, ignore_index=True)
        
    def append_D_S(self, new_sample):
        self.D_S_loss_df = self.D_S_loss_df.append(new_sample, ignore_index=True)
        
    def append_D_O(self, new_sample):
        self.D_O_loss_df = self.D_O_loss_df.append(new_sample, ignore_index=True)
        
    def append_SN_loss(self, new_sample):
        self.SN_loss_df = self.SN_loss_df.append(new_sample, ignore_index=True)

    def append_loss_step(self, new_sample):
        self.loss_step_df = self.loss_step_df.append(new_sample, ignore_index=True)

    def append_SN_acc(self, new_sample):
        self.SN_acc_df = self.SN_acc_df.append(new_sample, ignore_index=True)

    def append_acc_step(self, new_sample):
        self.acc_step_df = self.acc_step_df.append(new_sample, ignore_index=True)

    def save_logger(self, path, name, epoch=None):
        self.epoch = epoch
        if epoch is not None:
            file = os.path.join(path, str(self.epoch) + "_" + name + "_logger.pkl")
        else:
            file = os.path.join(path, name + "logger.pkl")
        pkl.dump(self, open(file, "wb"))

    def get_shades(self):
        """
        Create a colour palette for each base color. Base colour are:
        KEYS:
        - red
        - violet
        - blue
        - green
        - yellow
        - grey
        For each of them a list of
        :return: dict of list of self.shades shaded colour is created
        """
        self.shade_name = ['red', 'blue', 'yellow', 'green', 'violet', 'grey']
        red_list = color_hex_list_generator("#FF0000", "#F5B7B1", self.shades)
        viol_list = color_hex_list_generator("#4A235A", "#EBDEF0", self.shades)
        blue_list = color_hex_list_generator("#0B3AEA", "#8FF0FC", self.shades)
        green_list = color_hex_list_generator("#145A32", "#D1F2EB", self.shades)
        yellow_list = color_hex_list_generator("#FF6C00", "#F7DC6F", self.shades)
        grey_list = color_hex_list_generator("#17202A", "#F2F3F4", self.shades)
        color = {
            "red": red_list,
            "violet": viol_list,
            "blue": blue_list,
            "green": green_list,
            "yellow": yellow_list,
            "grey": grey_list,
        }
        return color

    def get_base_color(self):
        """
        Return a list of complementary colur
        :return: list of complementary color
        """
        color = [
            '#2971B0',  # "blue":
            '#3A9E1F',  # "green":
            '#E39112',  # "Orange":
            '#EC4040',  # "red":
            '#00FFFF',  # "light_blue"
            '#FFFF00',  # "yellow":
            '#FF00FF',  # "fuchsia":
            '#00FF00',  # "light_green":
            '#000000',  # "black":
            '#757575',  # "grey":
            '#7800FF',  # "violet":
        ]
        return color

    def create_figure(self):
        """
        Create a figure and add two plot to it
        :return:
        """
        self.loss_step_df = self.norm_step(self.loss_step_df)
        #self.acc_step_df = self.norm_step(self.acc_step_df)
        if self.mode == "train":
            self.fig_G_loss = self.generate_plot(self.G_loss_df, self.loss_step_df)
            self.fig_G_loss = self.plot_layout(self.fig_G_loss, "Generator Losses")
            self.fig_D_S_loss = self.generate_plot(self.D_S_loss_df, self.loss_step_df)
            self.fig_D_S_loss = self.plot_layout(self.fig_D_S_loss, "SAR Discriminator Losses")
            self.fig_D_O_loss = self.generate_plot(self.D_O_loss_df, self.loss_step_df)
            self.fig_D_O_loss = self.plot_layout(self.fig_D_O_loss, "Optical Discriminator Losses")

        '''self.fig_SN_loss = self.generate_plot(self.SN_loss_df, self.loss_step_df)
        self.fig_SN_loss = self.plot_layout(self.fig_SN_loss, "Segmentation Network Losses")
        self.fig_SN_acc = self.generate_plot(self.SN_acc_df, self.acc_step_df)
        self.fig_SN_acc = self.plot_layout(self.fig_SN_acc, "Segmentation Accuracy", "Epochs", "Accuracy [%]")'''

    def generate_plot(self, df, df_x):
        """
        Generates a plot where each column is added as a singal
        :param df: is the y values
        :param df_x: is the x value
        :return:
        """
        fig = go.Figure()
        for i, col in enumerate(df.columns):
            color = dict(color=self.base_color[i])
            fig.add_trace(go.Scatter(x=df_x['step'], y=df[col], mode='lines+markers', line=color, name=col))
        return fig

    def create_subplot(self):
        if self.mode == "train":
            self.fig_GAN = self.generate_subplot(self.loss_step_df, self.loss_step_df, self.G_loss_df, self.D_S_loss_df, self.D_O_loss_df)
            self.fig_GAN = self.plot_layout(self.fig_GAN, "GAN Losses", "", "")
        self.fig_SN = self.generate_subplot(self.loss_step_df, self.acc_step_df, self.SN_loss_df, self.SN_acc_df)
        self.fig_SN = self.plot_layout(self.fig_SN, "Segmentation Network Performances", "", "")

    def generate_subplot(self, df_x1, df_x2, df1, df2, df3=None):
        if df3 is not None:
            row = 3
        else:
            row = 2

        fig = make_subplots(rows=row, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.02)

        for i, col in enumerate(df1.columns):
            color = dict(color=self.base_color[i])
            fig.add_trace(go.Scatter(x=df_x1['step'], y=df1[col], mode='lines+markers', line=color, name=col), row=1, col=1)

        for i, col in enumerate(df2.columns):
            color = dict(color=self.base_color[i])
            fig.add_trace(go.Scatter(x=df_x2['step'], y=df2[col], mode='lines+markers', line=color, name=col), row=2, col=1)

        if df3 is not None:
            for i, col in enumerate(df3.columns):
                color = dict(color=self.base_color[i])
                fig.add_trace(go.Scatter(x=df_x1['step'], y=df3[col], mode='lines+markers', line=color, name=col), row=3, col=1)
        return fig
    
    @staticmethod
    def plot_layout(fig, title, x_title="Epochs", y_title="Loss Value"):
        """

        :param fig: fig to which apply the layout
        :param title: title
        :param x_title: title position x
        :param y_title: title position y
        :return:
        """
        fig.update_layout(
            #showlegend=False,
            title={'text': title, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis_title=x_title,
            yaxis_title=y_title,
            legend_x=0.84,
            legend_y=0.01,
            font=dict(family="Times New Roman, monospace", size=29, color="Black"),
            legend=dict(title="Legend", bgcolor="White", bordercolor="Black", borderwidth=2)
        )
        fig.update_xaxes(showline=True, linewidth=0.5, linecolor='Black', mirror=True, range=[-0.3, 30.3])
        #TODO: range
        fig.update_yaxes(showline=True, linewidth=0.5, linecolor='Black', mirror=True, range=[0, 100],  tickmode = 'linear', dtick = 10)
        fig.update_xaxes(ticks="outside")
        fig.update_yaxes(ticks="outside")
        return fig

    def norm_step(self, df):
        """
        Normilize the step value by the number of epoch so that to have as x the number of epoch not the bumber of step
        :param df: input data
        :return:
        """
        mx = df['step'].iloc[-1]
        if self.epoch is not None:
            norm = mx/self.epoch
            for i, val in enumerate(df['step']):
                df['step'][i] = val/norm
        return df

    def save_fig(self, path=""):
        """
        Save figures
        :param path:
        :return:
        """
        self.fig_G_loss.write_html("/home/ale/Desktop/fig_G_loss.html")
        self.fig_D_S_loss.write_html("/home/ale/Desktop/fig_D_S_loss.html")
        self.fig_D_O_loss.write_html("/home/ale/Desktop/fig_D_O_loss.html")
        self.fig_SN_loss.write_html("/home/ale/Desktop/fig_SN_loss.html")
        self.fig_SN_acc.write_html("/home/ale/Desktop/fig_SN_acc.html")

    def filter_all(self, win, pad):
        """
        filter all loss function with mov mean
        :param win: mov mean filter win
        :param pad:
        :return:
        """
        if self.mode == "train":
            self.G_loss_df = self.filter_df(self.G_loss_df, win, pad)
            self.D_S_loss_df = self.filter_df(self.D_S_loss_df, win, pad)
            self.D_O_loss_df = self.filter_df(self.D_O_loss_df, win, pad)
        self.SN_loss_df = self.filter_df(self.SN_loss_df, win, pad)
        # self.SN_acc_df = self.filter_df(self.SN_acc_df, win, pad)

    @staticmethod
    def filter_df(df, win, pad):
        """
        Aplly movmean
        :param df: data to be filtered
        :param win: win size
        :param pad:
        :return:
        """
        for i in df.columns:
            df[i] = moving_average(df[i], win, pad)
        return df

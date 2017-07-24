import numpy as np
import matplotlib
from kivy.app import App
from kivy.uix.slider import Slider
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.graphics.texture import Texture
from scipy import io
from scipy.misc import imread
from skimage.restoration import unwrap_phase
from scipy.ndimage.filters import gaussian_filter

matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

"""
Almost all visual aspect of this app are defined within the image.kv file, in this file we still create popup used for
selecting file to display as 3d map.
Global variable are used, fig, ax and surf_plot, can cause troubles when importing, but plt.gca() and plt.gcf() weren't
returning expected items.
"""

fig = ''
ax = ''
surf_plot = ''


def base_plot(ui, ccmap, samp=16):
    """
    Function used to generate 3d surface plot of given array. Values in array are treated as height of plot
    :param ui: should be a 2D numpy array
    :param ccmap: sting with name of colormap
    :param samp: used to increase sample rate of bigger arrays, smaller value, will increase lag of interface
    :return: Return 3 values, s is the canvas converted to rgb string, x,y are size of canvas.
    """
    global fig
    global ax
    global surf_plot
    fig = Figure(facecolor='#FAF0E6')
    ax = fig.gca(projection='3d', facecolor='#FAF0E6')
    ax._axis3don = False  # disable 3d axis on plot
    fig.set_dpi(400) # with this we should render canvas of size 3200x2400, should be enough even for 4k display :)
    # canvas is not write anywhere, and this settings don't decrease speed significantly.
    y1, x1 = ui.shape
    ax.auto_scale_xyz([0, x1], [0, x1], [0, 15])
    ccount = x1 / samp if x1 > samp * 50 else 50
    rcount = y1 / samp if y1 > samp * 50 else 50  # bigger arrays can have too small resolution on default sampling
    x1 = np.arange(x1)
    y1 = np.arange(y1)
    x1, y1 = np.meshgrid(x1, y1)
    surf_plot = ax.plot_surface(y1, x1, ui, cmap=ccmap, linewidth=0, antialiased=False, rcount=rcount, ccount=ccount)
    fig.colorbar(surf_plot, shrink=.5, label='Phase')
    fig.tight_layout()  # decrease size of margin around surface plot
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s = canvas.tostring_rgb()  # with s we update texture of displayed image
    x, y = fig.get_size_inches() * fig.dpi
    return s, x, y


def change_angle(azim=-60, elev=30):
    """
    Function to change view angles of current plot
    Change view angles of surface
    :param azim: azimuth of camera
    :param elev: elevation of camera
    :return: Return 3 values, s is the canvas converted to rgb string, x,y are size of canvas.
    """
    ax.azim = azim
    ax.elev = elev
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s = canvas.tostring_rgb()
    x, y = fig.get_size_inches() * fig.dpi
    return s, x, y


def change_colormap(spinner, ccmap):
    """
    Used to change colormap of current surface plot, it actually change the texture of image, other just prepare texture
    :param spinner: not used, bind method of spinner just tell what instance of spinner call this function
    :param ccmap: string with name of colormap
    """
    if surf_plot == '':
        return  # let make sure that user can change spinner before loading file, without crashing
    surf_plot.set_cmap(ccmap)  # just hack to change cmap
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s = canvas.tostring_rgb()
    x, y = fig.get_size_inches() * fig.dpi
    tex = Texture.create(size=(x, y), colorfmt='rgb')
    tex.blit_buffer(s, bufferfmt="ubyte", colorfmt="rgb")
    tex.flip_vertical()  # flip texture, otherwise it is upside down
    app = App.get_running_app()  # get instance of current app
    root = app.root  # get root widget of app
    root.ids.surf_image.texture = tex  # change the texture of image


class SpecSlider(Slider):
    """
    Simple class to override default on_touch_up method of slider
    """

    def on_touch_up(self, touch):
        """
        Override on_touch_up method of Slider class, otherwise is hard to make changes, only when we stop using slider,
        and with default events/binds it's only possible with each change of value.
        It handle event, and it will change the image texture within app.
        :param touch:
        :return:
        """
        if touch.grab_current is not self:
            return
        touch.ungrab(self)  # without it, app will not respond to new touches
        app = App.get_running_app()
        root = app.root
        azim = root.ids.azimuth_slider.value
        elev = root.ids.elevation_slider.value  # get value of two slider used in app
        s, x, y = change_angle(azim, elev)
        tex = Texture.create(size=(x, y), colorfmt='rgb')
        tex.blit_buffer(s, bufferfmt="ubyte", colorfmt="rgb")
        tex.flip_vertical()
        root.ids.surf_image.texture = tex


class LoadDialog(FloatLayout):
    """
    Simple class used to make popup with FileBrowser widget
    """
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class MainWidget(Widget):
    """
    Main Widget of app, handle button calls within app.
    """

    def __init__(self):
        """
        Bind Spinner with change_colormap function, so when user change value of spinner, application will call
        bind function to change colormap of plot
        """
        super(MainWidget, self).__init__()
        spinner = self.ids.cmap_spinner
        spinner.bind(text=change_colormap)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        """
        Create and open popup with filebrowser widget, and bind events of filebrowser widget with methods from this class.
        """
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(1, 1))
        self._popup.open()

    def load(self, file2open):
        """
        Open selected file, it handle .mat files and basic graphical files. Not nice approach using file extension.
        it prepare file, change to 2D array, smooth data with gaussian_filter
        :param file2open: path to file to open, with it filename.
        :return:
        """
        # todo handle unsupported files types, and make popup to choose again.
        if file2open[-4:].lower() == '.mat':
            surf = io.loadmat(file2open)
            surf = surf['u']
            surf = np.angle(surf)
            surf = unwrap_phase(surf)
        elif file2open[-4:].lower() == '.jpg' or file2open[-4:].lower() == '.png' or file2open[-4:].lower() == '.bmp':
            surf = imread(file2open, flatten=1).astype(np.float32)
        surf = gaussian_filter(surf, 10)  # some magic number for smoothing data :/
        s, x, y = base_plot(surf, self.ids.cmap_spinner.text)
        tex = Texture.create(size=(x, y), colorfmt='rgb')
        tex.blit_buffer(s, bufferfmt="ubyte", colorfmt="rgb")
        tex.flip_vertical()
        self.ids.surf_image.texture = tex
        self.ids.elevation_slider.value = 30
        self.ids.azimuth_slider.value = -60  # set azimuth and elevation slider value
        self.ids.elevation_slider.disabled = False
        self.ids.azimuth_slider.disabled = False  # make slider active, if they were disabled before.
        self.dismiss_popup()  # close popup with filebrowser widget


class ImageApp(App):
    """
    Main Class, it just call main widget and return it, and disable kivy pannel in settings
    """
    use_kivy_settings = False  # we don't need this pannel, but option still can be called with F1 key on PC machines

    def build(self):
        """
        Let me build this app
        """
        image = MainWidget()
        return image


if __name__ == '__main__':
    ImageApp().run()

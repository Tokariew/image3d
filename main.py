import numpy as np
import matplotlib
from kivy.app import App
from kivy.uix.slider import Slider
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.graphics.texture import Texture
from kivy.uix.settings import SettingsWithSidebar
from scipy import io
from scipy.misc import imread
from skimage.restoration import unwrap_phase
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

"""
Almost all visual aspect of this app are defined within the image.kv file, in this file we still create popup used for
selecting file to display as 3d map.
Global variable are used, fig, ax and surf_plot, can cause troubles when importing, but plt.gca() and plt.gcf() weren't
returning expected items.
"""
surf_plot = ''

json = """
[
  {
    "desc": "Enter sigma for gaussian filter used to smooth data. Bigger value produce smoother data",
    "key": "sigma",
    "section": "Misc",
    "title": "sigma",
    "type": "numeric"
  },
  {
    "desc": "Used as sampling factor, only one in this value row/column will be plotted. Smaller values increase processing time, but plot will look better.",
    "key": "sampling",
    "section": "Misc",
    "title": "sampling",
    "type": "numeric"
  }
]
"""


def base_plot(ui, ccmap, samp=16):
    """
    Function used to generate 3d surface plot of given array. Values in array are treated as height of plot
    :param ui: should be a 2D numpy array
    :param ccmap: sting with name of colormap
    :param samp: used to increase sample rate of bigger arrays, smaller value, will increase lag of interface
    :return: Return 3 values, s is the canvas converted to rgb string, x,y are size of canvas.
    """

    global surf_plot
    fig = plt.figure(facecolor='#FAF0E6')
    ax = fig.gca(projection='3d', facecolor='#FAF0E6')
    ax._axis3don = False  # disable 3d axis on plot
    fig.set_dpi(400)  # with this we should render canvas of size 3200x2400, should be enough even for 4k display :)
    # we don't save/write canvas on disk, and this settings don't decrease speed significantly.
    y1, x1 = ui.shape
    u_max = ui.max()
    ax.auto_scale_xyz([0, x1], [0, x1], [0, u_max])  # set scale of axis in plot
    ccount = x1 / samp if x1 > samp * 50 else 50
    rcount = y1 / samp if y1 > samp * 50 else 50  # bigger arrays can have too small resolution on default sampling
    x1 = np.arange(x1)
    y1 = np.arange(y1)
    x1, y1 = np.meshgrid(x1, y1)
    surf_plot = ax.plot_surface(y1, x1, ui, cmap=ccmap, linewidth=0, antialiased=False, rcount=rcount, ccount=ccount)
    plt.colorbar(surf_plot, shrink=.5, label='Phase')
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
    ax = plt.gca()
    ax.azim = azim
    ax.elev = elev
    fig = plt.gcf()
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
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s = canvas.tostring_rgb()
    x, y = fig.get_size_inches() * fig.dpi
    app = App.get_running_app()  # get instance of current app
    root = app.root  # get root widget of app
    root.make_texture(s, x, y)


class SpecSlider(Slider):
    """
    Simple class to override default on_touch_up method of slider
    """

    def on_touch_up(self, touch):
        """
        Override on_touch_up method of Slider class, otherwise is hard to make changes, only when we stop using slider,
        and with default events/binds it's only possible with each change of value.
        It handle event, and it will change the image texture within app.
        """
        if touch.grab_current is not self:
            return
        touch.ungrab(self)  # without it, app will not respond to new touches
        app = App.get_running_app()
        root = app.root
        azim = root.ids.azimuth_slider.value
        elev = root.ids.elevation_slider.value  # get value of two slider used in app
        s, x, y = change_angle(azim, elev)
        root.make_texture(s, x, y)


class LoadDialog(BoxLayout):
    """
    Simple class used to make popup with File Browser widget
    """
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class WrongFileDialog(Popup):
    """
    Class to display popup when user try to open unsupported file. Appearance is defined in kv file
    """

    def reload(self):
        """
        It's called when user press ok button, on popup. It's closed this popup, and reload LoadDialog popup.
        """
        self.dismiss()
        app = App.get_running_app()  # get instance of current app
        root = app.root
        root.popup.open()


class SelectMatVariable(Popup):
    """
    Class to display popup when loading .mat file with many variables
    """
    values = ObjectProperty(None)


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
        self.content = LoadDialog(load=self.open_file, cancel=self.dismiss_popup)
        self.popup = Popup(title="Load file", content=self.content, size_hint=(1, 1))
        self.popup2 = WrongFileDialog()
        self.popup3 = ''
        self.file2open = ''
        self.variable_tuple = ''
        self.mat_structure = ''

    def dismiss_popup(self):
        self.popup.dismiss()

    def show_load(self):
        """
        Create and open popup with file browser widget, and bind events of file browser widget with methods from this
        class.
        """
        self.popup.open()

    def mat_callback(self, instance):
        """
        called when user closed the popup in which he selected variable from matlab file. Load matlab file, load
        selected variable and call prepare_data
        """
        surf = io.loadmat(self.file2open)
        surf = surf[self.popup3.ids.mat_spinner.text]
        surf = np.angle(surf)
        surf = unwrap_phase(surf)
        self.prepare_data(surf)

    def open_file(self, file2open):
        """
        Open selected file, it handle .mat files and basic graphical files. Not nice approach using file extension.
        after file was open it call prepare_data method to handle open popup and prepare data to display
        :param file2open: path to file to open, with it filename.
        """
        if file2open[-4:].lower() == '.mat':
            self.mat_structure = io.whosmat(file2open)
            if len(self.mat_structure) == 1:
                surf = io.loadmat(file2open)
                surf = surf[self.mat_structure[0][0]]
                surf = np.angle(surf)
                surf = unwrap_phase(surf)
                self.prepare_data(surf)
            if len(self.mat_structure) > 1:
                self.variable_tuple = tuple(x[0] for x in self.mat_structure)
                self.popup3 = SelectMatVariable(values=self.variable_tuple)
                self.popup3.open()
                self.file2open = file2open
                self.popup3.bind(on_dismiss=self.mat_callback)

        elif file2open[-4:].lower() == '.jpg' or file2open[-4:].lower() == '.bmp' or file2open[-4:].lower() == '.png':
            surf = imread(file2open, flatten=True).astype(np.float32)
            self.prepare_data(surf)
        else:
            self.dismiss_popup()
            self.popup2.open()

    def prepare_data(self, surf):
        """
        this method prepare data to display, it smooth loaded data with gaussian filter, then it call function to plot
        data, another to make texture for picture, and reset values of sliders to default. And the end it close
        file browser popup.
        :param surf: 2d array with loaded values from the file
        """
        app = App.get_running_app()
        sigma = int(app.config.getdefault('Misc', 'sigma', 0))
        surf = gaussian_filter(surf, sigma)  # some magic number for smoothing data :/
        sampling = int(app.config.getdefault('Misc', 'sampling', 0))
        s, x, y = base_plot(surf, self.ids.cmap_spinner.text, sampling)
        self.make_texture(s, x, y)
        self.ids.elevation_slider.value = 30
        self.ids.azimuth_slider.value = -60  # set azimuth and elevation slider value
        self.ids.elevation_slider.disabled = False
        self.ids.azimuth_slider.disabled = False  # make slider active, if they were disabled before.
        self.dismiss_popup()  # close popup with file browser widget'''

    def make_texture(self, s, x, y):
        tex = Texture.create(size=(x, y), colorfmt='rgb')
        tex.blit_buffer(s, bufferfmt="ubyte", colorfmt="rgb")
        tex.flip_vertical()
        self.ids.surf_image.texture = tex


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
        self.settings_cls = SettingsWithSidebar  # set settings class, used to change appearance of setting screen
        return image

    def build_config(self, config):
        """Set default option for settings"""
        config.setdefaults('Misc', {'sigma': 10, 'sampling': 16})

    def build_settings(self, settings):
        """Add settings panel to settings screen, based on json"""
        settings.add_json_panel('Misc', self.config, data=json)


if __name__ == '__main__':
    ImageApp().run()

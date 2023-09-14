from typing import Tuple

from geovoronoi import voronoi_regions_from_coords
import matplotlib
import numpy
from shapely.geometry import Polygon, Point

from cdl_eeg.models.region_based_pooling.region_splits.region_split_base import RegionSplitBase
from cdl_eeg.models.region_based_pooling.utils import RegionID, project_to_2d, ChannelsInRegionSplit, ChannelsInRegion
from cdl_eeg.models.transformations.utils import UnivariateUniform


class VoronoiSplit(RegionSplitBase):
    # TODO: RuntimeWarning all over the place due to Polygons and MatplotlibDeprecationWarning on get_cmap

    __slots__ = "_points", "_voronoi"

    def __init__(self, num_points, x_min, x_max, y_min, y_max, method="box_uniform"):
        # ------------------
        # Sample the points
        # ------------------
        # TODO: This is often seen as poor programming practice... so I should probably change here
        if method == "box_uniform":
            self._points = _box_uniform(num_points, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        else:
            raise ValueError(f"Method '{method}' for sampling the points was not recognised")

        # ------------------
        # Compute the polygons
        # ------------------
        boundary = Polygon(((x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)))
        voronoi = voronoi_regions_from_coords(numpy.array(self._points), geo_shape=boundary)[0]

        self._voronoi = {RegionID(id_): polygon for id_, polygon in voronoi.items()}

    def place_in_regions(self, electrodes_3d):
        # Initialise the dict
        chs_in_regs = {region: [] for region in self.regions}

        # Project coordinates to 2D
        electrodes_2d = project_to_2d(electrodes_3d)

        # Loop through all electrodes
        for electrode_name, electrode_position in electrodes_2d.positions.items():
            # Try all polygons
            for polygon_id, polygon in self._voronoi.items():
                # If the 2D coordinate is contained in the current polygon, append it
                if polygon.contains(Point(electrode_position)):
                    chs_in_regs[polygon_id].append(electrode_name)
                    break
            else:
                # todo: consider raising warning. Do not print, it is just for current debugging
                print(f"Electrode {electrode_name} not contained in any polygon")

        # Return with correct type
        return ChannelsInRegionSplit({id_: ChannelsInRegion(tuple(ch_names)) for id_, ch_names in chs_in_regs.items()})

    def plot(self, face_color="random", edge_color="black", line_width=2, plot_seeds=False):
        _, ax = pyplot.subplots()
        for id_, area in self._voronoi.items():
            # Sample color from colormap
            cmap = matplotlib.colormaps.get_cmap('YlOrBr')

            # Get face color
            face_color = cmap(numpy.random.randint(low=0, high=cmap.N // 2))

            # Extract edges and plot (fill)
            x, y = area.exterior.xy
            ax.fill(x, y, linewidth=2, facecolor=face_color, edgecolor=edge_color)

        # Maybe plot the seed points as well
        if plot_seeds:
            x_vals, y_vals = zip(*self._points)
            ax.scatter(x_vals, y_vals)

    @property
    def regions(self) -> Tuple[RegionID, ...]:
        return tuple(self._voronoi.keys())


# ----------------
# Functions
# ----------------
def _box_uniform(k, *, x_min, x_max, y_min, y_max):
    """
    Samples k 2D coordiantes randomly (uniformly) from a box, bounded by the input arguments

    Parameters
    ----------
    k : int
        Number of points
    x_min : float
    x_max : float
    y_min : float
    y_max : float

    Returns
    -------
    tuple[tuple[float, float], ...]

    Examples
    --------
    >>> numpy.random.seed(1)
    >>> _box_uniform(3, x_min=0, x_max=1, y_min=-2, y_max=-1)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    ((0.4..., -1.2...), (0.0..., -1.6...), (0.1..., -1.9...))
    """
    # Create distributions
    x = UnivariateUniform(lower=x_min, upper=x_max)
    y = UnivariateUniform(lower=y_min, upper=y_max)

    # Sample and return
    return tuple((x.draw(), y.draw()) for _ in range(k))


if __name__ == "__main__":
    import mne
    from matplotlib import pyplot

    from cdl_eeg.models.region_based_pooling.utils import Electrodes3D

    # Create regions
    vor = VoronoiSplit(num_points=32, x_min=-.17, x_max=.17, y_min=-.17, y_max=.17)

    # Generate positions
    my_positions = Electrodes3D(mne.channels.make_standard_montage(kind="GSN-HydroCel-129").get_positions()["ch_pos"])

    # Place them
    placed_channels = vor.place_in_regions(my_positions)
    print(placed_channels)

    # Plot regions
    vor.plot()

    # Plot the channels
    for channel_names in placed_channels.ch_names.values():
        if len(channel_names) == 0:
            # Do not scatter plot empty regions
            continue

        positions_3d = Electrodes3D({ch_name: position for ch_name, position in my_positions.positions.items()
                                     if ch_name in channel_names.ch_names})
        pyplot.scatter(*zip(*project_to_2d(positions_3d).positions.values()))

    pyplot.show()

from geovoronoi import voronoi_regions_from_coords
import matplotlib
import numpy
from shapely.geometry import Polygon

from cdl_eeg.models.region_based_pooling.region_splits.region_split_base import RegionSplitBase
from cdl_eeg.models.transformations.utils import UnivariateUniform


class VoronoiSplit(RegionSplitBase):
    # TODO: RuntimeWarning all over the place due to Polygons and MatplotlibDeprecationWarning on get_cmap
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
        # Compute the boundaries
        # ------------------
        shape_boundary = Polygon(((x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)))
        self._voronoi = voronoi_regions_from_coords(numpy.array(self._points), geo_shape=shape_boundary)[0]

    def place_in_regions(self, electrode_positions):
        raise NotImplementedError

    def plot(self):
        _, ax = pyplot.subplots()
        for area in self._voronoi.values():
            # Sample color from colormap
            cmap = matplotlib.cm.get_cmap('YlOrBr')

            # Get face color
            face_color = cmap(numpy.random.randint(low=0, high=cmap.N // 2))
            x, y = area.exterior.xy
            ax.fill(x, y, linewidth=2, facecolor=face_color, edgecolor="black")

        x_vals = tuple(p[0] for p in self._points)
        y_vals = tuple(p[1] for p in self._points)
        ax.scatter(x_vals, y_vals)


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
    from matplotlib import pyplot

    vor = VoronoiSplit(num_points=11, x_min=0, x_max=1, y_min=-2, y_max=-1)
    vor.plot()
    pyplot.show()
    # print(vor._voronoi.ridge_vertices)

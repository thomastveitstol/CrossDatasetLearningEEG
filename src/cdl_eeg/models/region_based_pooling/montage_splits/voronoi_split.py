from typing import Dict, List, Tuple

from geovoronoi import voronoi_regions_from_coords
import matplotlib
import numpy
from shapely.geometry import Polygon, Point

from cdl_eeg.data.datasets.getter import get_channel_system
from cdl_eeg.models.region_based_pooling.montage_splits.montage_split_base import MontageSplitBase
from cdl_eeg.models.region_based_pooling.utils import RegionID, project_to_2d, ChannelsInRegionSplit, ChannelsInRegion
from cdl_eeg.models.transformations.utils import UnivariateUniform


class VoronoiSplit(MontageSplitBase):
    # TODO: RuntimeWarning all over the place due to Polygons and MatplotlibDeprecationWarning on get_cmap

    __slots__ = "_voronoi"

    def __init__(self, channel_systems, min_nodes, x_min, x_max, y_min, y_max, num_initial_centroids=500):
        """
        Initialise

        Parameters
        ----------
        channel_systems : str | tuple[str, ...]
        min_nodes : int
        x_min : float
        x_max : float
        y_min : float
        y_max : float
        num_initial_centroids : int
        """
        # ------------------
        # Compute the polygons
        # ------------------
        self._voronoi = _pruned_random_centroids(channel_systems=channel_systems, min_nodes=min_nodes, x_min=x_min,
                                                 x_max=x_max, y_min=y_min, y_max=y_max,
                                                 num_initial_centroids=num_initial_centroids)

    def place_in_regions(self, electrodes_3d):
        # Initialise the dict
        chs_in_regs: Dict[RegionID, List[str]] = {region: [] for region in self.regions}

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

    def plot(self, face_color="random", edge_color="black", line_width=2):
        _, ax = pyplot.subplots()
        for id_, area in self._voronoi.items():
            # Sample color from colormap
            cmap = matplotlib.colormaps.get_cmap('YlOrBr')

            # Get face color
            face_color = cmap(numpy.random.randint(low=0, high=cmap.N // 2))

            # Extract edges and plot (fill)
            x, y = area.exterior.xy
            ax.fill(x, y, linewidth=2, facecolor=face_color, edgecolor=edge_color)

    @property
    def regions(self) -> Tuple[RegionID, ...]:
        return tuple(self._voronoi.keys())


# ----------------
# Functions
# ----------------
def _electrode_2d_to_tuple(electrodes_2d):
    """
    Convert from Electrodes2D to a tuple of Point2D (channel names are omitted)

    Parameters
    ----------
    electrodes_2d : cdl_eeg.models.region_based_pooling.utils.Electrodes2D

    Returns
    -------
    tuple[Point2D, ...]

    Examples
    --------
    >>> from cdl_eeg.models.region_based_pooling.utils import Electrodes2D
    >>> _electrode_2d_to_tuple(Electrodes2D({"a": (1, 2), "b": (6, 2)}))
    ((1, 2), (6, 2))
    """
    return tuple(tuple(pos) for pos in electrodes_2d.positions.values())


def _pruned_random_centroids(channel_systems, min_nodes, x_min, x_max, y_min, y_max, num_initial_centroids):
    """
    This function generates many centroid, for then to prune it to satisfy the criteria for minimum number of electrodes
    for all channel systems.

    todo: a little time-consuming

    Parameters
    ----------
    channel_systems : str | tuple[str, ...]
    min_nodes : int
    x_min : float
    x_max : float
    y_min : float
    y_max : float
    num_initial_centroids : int

    Returns
    -------
    dict[RegionID, Polygon]
    """
    # Input checks
    if not isinstance(min_nodes, int):
        raise TypeError(f"Expected minimum number of nodes in a region to be integer, but found {type(min_nodes)}")
    if min_nodes < 1:
        raise ValueError(f"Expected minimum number of nodes in a region to be equal or greater than 1, but found "
                         f"{min_nodes}")

    # Get channel positions
    channel_systems = (channel_systems,) if isinstance(channel_systems, str) else channel_systems
    channel_positions = tuple(
        _electrode_2d_to_tuple(project_to_2d(get_channel_system(channel_system).electrode_positions))
        for channel_system in channel_systems
    )

    # ------------------
    # Create first Voronoi split
    # ------------------
    # Sample centroids
    centroids = _box_uniform(k=num_initial_centroids, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    # Create polygons
    boundary = Polygon(((x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)))
    voronoi = voronoi_regions_from_coords(numpy.array(centroids), geo_shape=boundary)[0]
    vor_cells = {RegionID(id_): polygon for id_, polygon in voronoi.items()}

    # Store cell centroids
    cell_centroids = dict()
    for region_id, polygon in vor_cells.items():
        # Get centroid of the current polygon
        _centroid = tuple(centroid for centroid in centroids if polygon.contains(Point(centroid)))

        # Verify that there is only one match
        assert len(_centroid) == 1, (f"Expected only one centroid in the Voronoi cell, but found {len(_centroid)}. "
                                     f"This should never happen, please contact the developer")

        # Store the centroid
        cell_centroids[region_id] = tuple(centroid for centroid in centroids if polygon.contains(Point(centroid)))[0]

    # ------------------
    # Remove all centroid which does not contain
    # a single electrode for any channel system
    # ------------------
    # Loop through all channel systems
    empty_vor_cells = set()
    for positions in channel_positions:
        # Compute empty cells in this channel system
        empty = (region_id for region_id, polygon in vor_cells.items() if not any(polygon.contains(Point(point))
                                                                                  for point in positions))

        # Update the Voronoi cells to delete
        empty_vor_cells.update(empty)

    # Delete all empty cells and centroids
    for cell in empty_vor_cells:
        del vor_cells[cell]
        del cell_centroids[cell]

    # Recompute the Voronoi cells
    voronoi = voronoi_regions_from_coords(numpy.array(tuple(cell_centroids.values())), geo_shape=boundary)[0]
    vor_cells = {RegionID(id_): polygon for id_, polygon in voronoi.items()}

    # Return the voronoi cells if non-emptiness is the criteria
    if min_nodes == 1:
        return vor_cells

    # TODO: finish implementation for different stopping criteria
    raise NotImplementedError


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

    from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
    from cdl_eeg.models.region_based_pooling.utils import Electrodes3D

    # Create regions
    vor = VoronoiSplit(channel_systems=("YulinWang", "HatlestadHall", "Miltiadous"), min_nodes=1, x_min=-.17, x_max=.17,
                       y_min=-.17, y_max=.17)

    # Generate positions
    my_positions = Miltiadous().channel_system.electrode_positions

    # Place them
    placed_channels = vor.place_in_regions(my_positions)

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

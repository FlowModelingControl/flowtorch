"""Module with classes and functions to explore and analyse iPSP data.
"""

import torch as pt
import plotly.graph_objects as go
import plotly.express as px
from flowtorch.data import PSPDataloader
from plotly.subplots import make_subplots
from typing import List


class PSPExplorer(object):
    def __init__(self, file_path: str):
        self._file_path = file_path
        self._loader = PSPDataloader(file_path)

    def _aspect_ratio(self, vertices: pt.Tensor) -> dict:
        dx = abs((pt.max(vertices[:, :, 0]) -
                  pt.min(vertices[:, :, 0])).item())
        dy = abs((pt.max(vertices[:, :, 1]) -
                  pt.min(vertices[:, :, 1])).item())
        dz = abs((pt.max(vertices[:, :, 2]) -
                  pt.min(vertices[:, :, 2])).item())
        return {"x": 1.0, "y": dy/dx, "z": dz/dx}

    def _create_time_slider(self, times):
        steps = []
        for i in range(len(times)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(times)},
                      {"title": "Selected time: {:s}".format(times[i])}],
                label=str(i)
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)
        slider = dict(
            active=0,
            pad={"t": 50},
            steps=steps
        )
        return slider

    def _create_surface_trace(self, field, vertices, weights, every, cmin, cmax):
        return go.Surface(
                x=vertices[::every, ::every, 0],
                y=vertices[::every, ::every, 1],
                z=vertices[::every, ::every, 2],
                surfacecolor=field[::every, ::every] *
                weights[::every, ::every],
                cmin=cmin,
                cmax=cmax
            )

    def _create_surface_layout(self, vertices, width, times=None):
        aspect = self._aspect_ratio(vertices)
        if times is None:
            sliders = None
        else:
            sliders = [self._create_time_slider(times)]
        layout = go.Layout(
            width=width,
            height=int(width*aspect["x"]),
            scene={
                "camera_eye": {"x": 0, "y": -1.0, "z": 0.5},
                "aspectratio": aspect
            },
            sliders=sliders
        )
        return layout

    def animate(self, file_name, times):
        pass

    def interact(self, dataset: str, field_name: str, times: list, mask: bool = True,
                 width: int = 768, every: int = 5, cmin: float = -2, cmax: float = 0.5):
        self._loader.select_dataset(dataset)
        vertices = self._loader.get_vertices()
        weights = self._loader.get_weights()
        if not mask:
            weights = 1.0
        fields = self._loader.load_snapshot(field_name, times)
        layout = self._create_surface_layout(vertices, width, times)
        fig = go.Figure(layout=layout)
        for i in range(len(times)):
            field = fields[:, :, i]
            surface = self._create_surface_trace(field, vertices, weights, every, cmin, cmax)
            fig.add_trace(surface)
        fig.show()

    def mean(self, dataset: str, field_name: str, times: list,
                mask: bool=True, width: int = 1024, every=5,
                cmin: float=-2, cmax: float = 0.5):
        vertices = self._loader.get_vertices()
        weights = self._loader.get_weights()
        if not mask:
            weights = 1.0
        fields = self._loader.load_snapshot(field_name, times)
        mean = pt.mean(fields, dim=2)
        layout = self._create_surface_layout(vertices, width)
        fig = go.Figure(layout=layout)
        surface = self._create_surface_trace(mean, vertices, weights, every, cmin, cmax)
        fig.add_trace(surface)
        fig.show()

    def std(self, dataset: str, field_name: str, times: list,
                mask: bool=True, width: int = 1024, every=5,
                cmin: float=0, cmax: float = 0.5):
        vertices = self._loader.get_vertices()
        weights = self._loader.get_weights()
        if not mask:
            weights = 1.0
        fields = self._loader.load_snapshot(field_name, times)
        std = pt.std(fields, dim=2)
        layout = self._create_surface_layout(vertices, width)
        fig = go.Figure(layout=layout)
        surface = self._create_surface_trace(std, vertices, weights, every, cmin, cmax)
        fig.add_trace(surface)
        fig.show()

    def write_times(self):
        return self._loader.write_times()

    def field_names(self):
        return self._loader.field_names()

    def datasets(self):
        return self._loader._dataset_names

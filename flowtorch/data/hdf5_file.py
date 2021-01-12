r"""Module to read and write the internal flowTorch data format.

The :class:`HDF5Writer` class allows to write field and mesh data
into an HDF5 file. It also creates XDMF files for postprocessing
in ParaView. The :class:`HDF5Dataloader` is a concrete :class:`Dataloader`
class that allows efficient batch access to simulation data stored in the
internal flowTorch data format.
"""

from .dataloader import Dataloader
from .foam_dataloader import FOAMCase, FOAMMesh, FOAMDataloader, POLYMESH_PATH, MAX_LINE_HEADER, FIELD_TYPE_DIMENSION
from .mpi_tools import main_only, main_bcast, job_conditional, log_message
from os.path import exists
from os import remove
from h5py import File
from mpi4py import MPI
import torch as pt


CONST_GROUP = "constant"
VAR_GROUP = "variable"
VERTICES_DS = "vertices"
CONNECTIVITY_DS = "connectivity"
CENTERS_DS = "centers"
VOLUMES_DS = "volumes"
XDMF_HEADER = """<source lang="xml" line="1">
    <Domain>
        <Grid Name="flowTorch" GridType="Collection" CollectionType="Temporal">
"""
XDMF_FOOTER = """
        </Grid>
    </Domain>
</source> 
"""
TOPOLOGY = "topology"
GEOMETRY = "geometry"

dtype_conversion = {
    pt.float32: "f4",
    pt.float64: "f8",
    pt.int32: "i4",
    pt.int64: "i8",
    "float32": "f4",
    "float64": "f8",
    "int32": "i4",
    "int64": "i8"
}

xdmf_attributes = {
    1: "Scalar",
    2: "Vector",
    3: "Vector",
    6: "Tensor6",
    9: "Tensor"
}

xdmf_dtype_conversion = {
    "f4": ("Float", 4),
    "f8": ("Float", 8),
    "i4": ("Int", 4),
    "i8": ("Int", 8),
    "float32": ("Float", 4),
    "float64": ("Float", 8),
    "int32": ("Int", 4),
    "int64": ("Int", 8)
}


class HDF5Dataloader(Dataloader):
    """
    """

    def __init__(self, file: str, dtype: str = pt.float32):
        """

        :param path: [description]
        :type path: [type]
        :param file_name: [description], defaults to FILE_NAME
        :type file_name: [type], optional
        :param dtype: [description], defaults to pt.float32
        :type dtype: [type], optional
        """
        pass

    def write_times(self):
        pass

    def field_names(self):
        pass

    def load_snapshot(self):
        pass

    def get_vertices(self):
        pass

    def get_weights(self):
        pass


class HDF5Writer(object):
    """Class to write flowTorch data to HDF5 file.

    Two types of data are supported:
    - variable: (field) data that changes with times, e.g, snapshots
    - constant: constant data like mesh vertices or cell volumes

    A XDMF accessor file can be created to support visual post-processing
    with ParaView and other XDMF-compatible software packages.
    """

    def __init__(self, file: str):
        """Construct :class:`HDF5Writer` object based on file name.

        :param file: path and file name to HDF5 file.
        :type file: str
        """
        self._file_path = file
        self._file = File(file, mode="a", driver="mpio", comm=MPI.COMM_WORLD)

    def __del__(self):
        """Destructor to ensure that HDF5 file is closed.
        """
        self._file.close()

    def write(self,
              name: str,
              size: tuple,
              data: pt.Tensor = None,
              time: str = None,
              dtype: str = pt.float32
              ):
        """Write data to HDF5 file.

        :param name: dataset name
        :type name: str
        :param size: dataset shape
        :type size: tuple
        :param data: data to write; if None, dataset is only allocated
        :type data: pt.Tensor, optional
        :param time: snapshot time, dataset if created in VAR_GROUP if present
        :type time: str, optional
        :param dtype: data type, defaults to pt.float32
        :type dtype: str, optional
        """
        if time is not None:
            ds_name = VAR_GROUP + "/{:s}/{:s}".format(time, name)
        else:
            ds_name = CONST_GROUP + "/{:s}".format(name)

        if dtype in dtype_conversion.keys():
            if ds_name in self._file:
                del self._file[ds_name]

            ds = self._file.create_dataset(
                ds_name,
                size,
                dtype=dtype_conversion[dtype]
            )
            if data is not None:
                shape_diff = len(size) - len(data.size())
                if shape_diff == 1:
                    data = data.unsqueeze(-1)
                elif shape_diff == -1:
                    data = data.squeeze()
                ds[:] = data.numpy()
        else:
            print(
                "Warning: invalid data type {:s} for field {:s}. Skipping field.".format(
                    str(dtype), name)
            )

    @main_only
    def write_xdmf(self):
        writer = XDMFWriter(self._file_path, self._file)
        writer.create_xdmf("flowtorch.xdmf")


class FOAM2HDF5(object):
    def __init__(self, path, dtype=pt.float32):
        self._loader = FOAMDataloader(path)
        self._dtype = dtype
        self._topology = None
        self._mesh_points = None

    def convert(self, filename, skip_zero=True):
        """Convert OpenFOAM case to flowTorch HDF5 file.

        :param filename: name of the HDF5 file
        :type filename: str
        :param skip_zero: skip zero folder if true; defaults to True
        :type skip_zero: bool, optional
        """
        file_path = self._loader._case._path + "/" + filename
        self._remove_file_if_present(file_path)
        if self._loader._case._distributed:
            message = """The direct conversion of distributed cases is currently not supported.\n
Workaround:
    1. run reconstructPar (OpenFOAM utility)
    1.1 if there is no reconstructed mesh, run also reconstructParMesh
    2. remove all processor* folders
    3. perform the conversion again (flowTorch)
            """
            log_message(message)
        else:
            log_message("Writing data to file {:s}".format(file_path))
            writer = HDF5Writer(file_path)
            log_message("Converting mesh.")
            self._convert_mesh(writer)
            log_message("Converting fields.")
            self._convert_fields(writer, skip_zero)
            log_message("Conversion finished.")
            writer.write_xdmf()

    @main_only
    def _remove_file_if_present(self, file_path):
        """Remove output file from previous runs if present

        :param file_path: path to file
        :type file_path: str
        """
        if exists(file_path):
            print("Removing old file {:s}".format(file_path))
            remove(file_path)

    def _convert_mesh(self, writer):
        mesh_path = self._loader._case._path + "/" + POLYMESH_PATH
        n_cells, n_points, n_top = self._gather_mesh_information(mesh_path)
        data = self._get_vertices(mesh_path, job=0)
        writer.write(VERTICES_DS, (n_points, 3), data, None, self._dtype)
        data = self._get_topology(job=0)
        writer.write(CONNECTIVITY_DS, (n_top,), data, None, pt.int32)
        data = self._get_cell_centers(job=1)
        writer.write(CENTERS_DS, (n_cells, 3), data, None, self._dtype)
        data = self._get_cell_volumes(job=1)
        writer.write(VOLUMES_DS, (n_cells,), data, None, self._dtype)

    @main_bcast
    def _gather_mesh_information(self, mesh_path):
        """Gather information for parallel writing of mesh data.


        :param mesh_path: [description]
        :type mesh_path: [type]
        """
        n_cells = self._loader._mesh._get_n_cells(mesh_path)
        n_points_faces, faces = self._loader._mesh._parse_faces(mesh_path)
        owners, neighbors = self._loader._mesh._parse_owners_and_neighbors(
            mesh_path)
        self._mesh_points = self._loader._mesh._parse_points(mesh_path)

        cell_faces = [[] for _ in range(n_cells)]
        n_faces_cell = pt.zeros(n_cells, dtype=pt.int32)
        n_face_labels = 0

        for i, owner in enumerate(owners):
            cell_faces[owner].append(faces[i][:n_points_faces[i]])
            n_face_labels += n_points_faces[i]
            n_faces_cell[owner] += 1
        for i, neigh in enumerate(neighbors):
            cell_faces[neigh].append(faces[i][:n_points_faces[i]])
            n_face_labels += n_points_faces[i]
            n_faces_cell[neigh] += 1

        topology_length = n_cells * 2 + \
            pt.sum(n_faces_cell).item() + n_face_labels
        self._topology = pt.zeros(topology_length, dtype=pt.int32)
        marker = 0
        for i, faces in enumerate(cell_faces):
            self._topology[marker] = 16
            self._topology[marker+1] = len(faces)
            marker += 2
            for j in range(len(faces)):
                n_labels = faces[j].size()[0]
                self._topology[marker] = n_labels
                self._topology[marker+1:marker+1+n_labels] = faces[j]
                marker += n_labels + 1
        return n_cells, self._mesh_points.size()[0], self._topology.size()[0]

    @job_conditional
    def _get_topology(self, job=0):
        return self._topology

    @job_conditional
    def _get_vertices(self, mesh_path, job=0):
        return self._mesh_points

    @job_conditional
    def _get_cell_centers(self, job=0):
        return self._loader._mesh.get_cell_centers()

    @job_conditional
    def _get_cell_volumes(self, job=0):
        return self._loader._mesh.get_cell_volumes()

    def _convert_fields(self, writer, skip_zero):
        """Convert convert OpenFOAM fields to HDF5.

        :param writer: HDF5 writer
        :type writer: :class:`HDF5Writer`
        :param skip_zero: skip zero folder if true
        :type skip_zero: bool
        """
        field_info = self._gather_field_information(skip_zero)
        for job, info in enumerate(field_info):
            data = self._load_field(*info[:2], job=job)
            writer.write(info[0], info[2], data, info[1])

    @main_bcast
    def _gather_field_information(self, skip_zero):
        """Gather field information for parallel writing.

        - check if field type is supported
        - determine data size

        :param skip_zero: skip zero folder if true
        :type skip_zero: bool
        :return: list of all fields; each list element is a list
            with the entries [name, time, shape]
        :rtype: list
        """
        def load_n_lines(file_name, n):
            lines = []
            with open(file_name, "rb") as file:
                for _ in range(n):
                    lines.append(file.readline())
            return lines

        field_info = []
        mesh_path = self._loader._case._path + "/" + POLYMESH_PATH
        n_cells = self._loader._mesh._get_n_cells(mesh_path)
        all_fields = self._loader.field_names()
        if skip_zero and "0" in all_fields.keys():
            del all_fields["0"]
        for time in all_fields.keys():
            for name in all_fields[time]:
                path = self._loader._case.build_file_path(name, time)
                header = load_n_lines(path, MAX_LINE_HEADER)
                field_type = self._loader._field_type(header)
                if field_type in FIELD_TYPE_DIMENSION.keys():
                    field_info.append(
                        [name, time, (n_cells, FIELD_TYPE_DIMENSION[field_type])])
        return field_info

    @job_conditional
    def _load_field(self, field, time, job=0):
        return self._loader.load_snapshot(field, time)


class XDMFWriter(object):
    def __init__(self, file_path: str, hdf5_file: File):
        if "/" in file_path:
            self._path = file_path[:file_path.rfind("/")]
            self._hdf5_filename = file_path[file_path.rfind("/") + 1:]
        else:
            self._path = "."
            self._hdf5_filename = file_path
        self._file = hdf5_file
        self._n_cells = self._get_n_cells()

    @classmethod
    def from_filepath(cls, file_path: str):
        return cls(
            file_path,
            File(file_path, mode="a", driver="mpio", comm=MPI.COMM_WORLD)
        )

    def _get_n_cells(self) -> int:
        n_cells = 0
        location = "/{:s}/{:s}".format(CONST_GROUP, VOLUMES_DS)
        if location in self._file:
            n_cells = self._file[location].shape[0]
        if n_cells == 0:
            log_message("XDMF warning: could not determine number of cells.")
        return n_cells

    def _add_grid(self, time: str, offset: str = "") -> str:
        """
        """
        grid = offset + "<Grid Name=\"Grid\" GridType=\"Uniform\">\n"
        grid += self._add_topology(offset + " "*4)
        grid += self._add_geometry(offset + " "*4)
        if time is not None:
            grid += offset + " "*4 + "<Time Value=\"{:s}\"/>\n".format(time)
            for key in self._find_attributes(time):
                grid += self._add_attribute(time, key, offset + " "*4)
        grid += offset + "</Grid>\n"
        return grid

    def _find_attributes(self, time: str) -> list[str]:
        location = "/{:s}/{:s}".format(VAR_GROUP, time)
        keys = self._file[location].keys()
        valid_attr = []
        for key in keys:
            loc = location + "/{:s}".format(key)
            first_dim = self._file[loc].shape[0]
            if first_dim == self._n_cells:
                valid_attr.append(key)
        return valid_attr

    def _add_topology(self, offset: str = "") -> str:
        topology = offset + \
            "<Topology Name=\"{:s}\" TopologyType=\"Mixed\">\n".format(
                TOPOLOGY)
        location = self._hdf5_filename + \
            ":/{:s}/{:s}".format(CONST_GROUP, CONNECTIVITY_DS)
        topology += self._add_dataitem(location, offset + " "*4)
        topology += offset + "</Topology>\n"
        return topology

    def _add_geometry(self, offset: str = "") -> str:
        geometry = offset + "<Geometry GeometryType=\"XYZ\">\n"
        location = self._hdf5_filename + \
            ":/{:s}/{:s}".format(CONST_GROUP, VERTICES_DS)
        geometry += self._add_dataitem(location, offset + " "*4)
        geometry += offset + "</Geometry>\n"
        return geometry

    def _add_attribute(self, time: str, name: str, offset: str = "") -> str:
        location = self._hdf5_filename + ":/{:s}/{:s}/{:s}".format(
            VAR_GROUP, time, name
        )
        shape = self._file[location.split(":")[-1]].shape
        tensor_type = xdmf_attributes[len(shape)]
        attribute = offset + "<Attribute Name=\"{:s}\" AttributeType=\"{:s}\" Center=\"Cell\">\n".format(
            name, tensor_type
        )
        attribute += self._add_dataitem(location, offset + " "*4)
        attribute += offset + "</Attribute>\n"
        return attribute

    def _add_dataitem(self, location: str,  offset: str = "") -> str:
        path_in_file = location.split(":")[-1]
        shape = self._file[path_in_file].shape
        dimensions = " ".join(["{:d}".format(i) for i in shape])
        dtype, precision = xdmf_dtype_conversion[
            str(self._file[path_in_file].dtype)
        ]
        dataitem = offset + "<DataItem Dimensions=\"{:s}\" NumberType=\"{:s}\" Precision=\"{:d}\" Format=\"HDF\">\n".format(
            dimensions, dtype, precision
        )
        dataitem += offset + " "*4 + "{:s}\n".format(location)
        dataitem += offset + "</DataItem>\n"
        return dataitem

    def create_xdmf(self, filename: str = None):
        """

        :param filename: [description]
        :type filename: [type]
        """
        xdmf_str = XDMF_HEADER
        times = list(self._file[VAR_GROUP].keys())
        if len(times) > 0:
            for time in times:
                xdmf_str += self._add_grid(time, " "*12)
        else:
            xdmf_str += self._add_grid(None, " "*12)
        xdmf_str = xdmf_str[:-1]  # remove last linebreak
        xdmf_str += XDMF_FOOTER

        if filename is None:
            if "." in self._hdf5_filename:
                filename = self._hdf5_filename[:self._hdf5_filename.rfind(
                    ".")] + ".xdmf"
            else:
                filename = self._hdf5_filename + ".xdmf"
        log_message(
            "Writing file {:s} as wrapper for {:s} at location {:s}".format(
                filename, self._hdf5_filename, self._path
            )
        )
        with open(self._path + "/" + filename, "w") as file:
            file.write(xdmf_str)

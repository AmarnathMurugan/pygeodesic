# distutils: language = c++
# -*- coding: utf-8 -*-
#!python
#cython: language_level=3

import numpy
cimport numpy
from libcpp.vector cimport vector
from libcpp cimport bool as cbool


# =============================================================================
# C++ extern declarations
# =============================================================================

cdef extern from "geodesic_kirsanov/geodesic_mesh_elements.h" namespace "geodesic":

    cdef cppclass MeshElementBase:
        unsigned& id()

    cdef cppclass Vertex:
        Vertex()
        double& x()
        double& y()
        double& z()
        unsigned& id()
        cbool& saddle_or_boundary()

    cdef cppclass Edge:
        Edge()
        unsigned& id()
        double& length()
        Vertex* v0()
        Vertex* v1()
        cbool is_boundary()

    cdef cppclass Face:
        Face()
        unsigned& id()
        double* corner_angles()

    cdef enum PointType:
        VERTEX,
        EDGE,
        FACE,
        UNDEFINED_POINT

    cdef cppclass SurfacePoint:
        SurfacePoint()
        SurfacePoint(Vertex*)
        double& x()
        double& y()
        double& z()
        unsigned base_element_id()
        PointType type()

cdef extern from "geodesic_kirsanov/geodesic_mesh.h" namespace "geodesic":
    cdef cppclass Mesh:
        Mesh()
        void initialize_mesh_data(vector[double]&, vector[unsigned]&)
        vector[Vertex]& vertices()
        vector[Edge]& edges()
        vector[Face]& faces()

cdef extern from "geodesic_kirsanov/geodesic_algorithm_exact.h" namespace "geodesic":
    cdef cppclass GeodesicAlgorithmExact:
        GeodesicAlgorithmExact(Mesh*)
        void propagate(vector[SurfacePoint]&, double, vector[SurfacePoint]*)
        void propagate(vector[SurfacePoint]&, double)
        unsigned best_source(SurfacePoint&, double&)
        void trace_back(SurfacePoint&, vector[SurfacePoint]&)
        void geodesic(SurfacePoint&, SurfacePoint&, vector[SurfacePoint]&)

cdef extern from "geodesic_kirsanov/geodesic_algorithm_base.h" namespace "geodesic":
    double length(vector[SurfacePoint]&)

cdef extern from "geodesic_kirsanov/geodesic_constants_and_simple_functions.h" namespace "geodesic":
    double GEODESIC_INF

# Helper functions for modifying mesh data
cdef extern from "geodesic_kirsanov/geodesic_helpers.h" namespace "geodesic":
    void set_edge_length(Mesh* mesh, unsigned edge_idx, double length)
    void set_corner_angle(Mesh* mesh, unsigned face_idx, unsigned corner_idx, double angle)
    void set_saddle_or_boundary(Mesh* mesh, unsigned vertex_idx, cbool value)
    double get_edge_length(Mesh* mesh, unsigned edge_idx)
    double get_corner_angle(Mesh* mesh, unsigned face_idx, unsigned corner_idx)
    cbool get_saddle_or_boundary(Mesh* mesh, unsigned vertex_idx)
    void get_edge_vertices(Mesh* mesh, unsigned edge_idx, unsigned* v0, unsigned* v1)
    cbool is_edge_boundary(Mesh* mesh, unsigned edge_idx)
    void get_face_edge_lengths(Mesh* mesh, unsigned face_idx, double* lengths)
    void get_face_vertex_ids(Mesh* mesh, unsigned face_idx, unsigned* vertex_ids)


# =============================================================================
# Python enum / lookup
# =============================================================================

_POINT_TYPE_NAMES = {
    VERTEX: "VERTEX",
    EDGE: "EDGE",
    FACE: "FACE",
    UNDEFINED_POINT: "UNDEFINED_POINT",
}


# =============================================================================
# PyVertex
# =============================================================================

cdef class PyVertex:
    """Python-side snapshot of a geodesic::Vertex."""

    cdef readonly unsigned id
    cdef readonly double x, y, z
    cdef readonly bint saddle_or_boundary

    def __cinit__(self):
        self.id = 0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.saddle_or_boundary = False

    @staticmethod
    cdef PyVertex from_cpp(Vertex* v, Mesh* mesh):
        """
        Build a PyVertex by reading the C++ Vertex and its adjacency
        lists.  *mesh* is passed so we can safely resolve the
        SimpleVector pointers that Vertex stores.
        """
        cdef PyVertex pv = PyVertex.__new__(PyVertex)
        cdef unsigned i

        pv.id = v.id()
        pv.x  = v.x()
        pv.y  = v.y()
        pv.z  = v.z()
        pv.saddle_or_boundary = v.saddle_or_boundary()
        return pv

    @property
    def coords(self):
        """Return (x, y, z) tuple."""
        return (self.x, self.y, self.z)

    def __repr__(self):
        return (
            f"PyVertex(id={self.id}, "
            f"coords=({self.x:.6g}, {self.y:.6g}, {self.z:.6g}), "
            f"saddle_or_boundary={self.saddle_or_boundary})"
        )


# =============================================================================
# PyEdge
# =============================================================================

cdef class PyEdge:
    """Python-side snapshot of a geodesic::Edge."""

    cdef readonly unsigned id
    cdef readonly double length
    cdef readonly unsigned v0_id, v1_id
    cdef readonly bint boundary

    def __cinit__(self):
        self.id = 0
        self.length = 0.0
        self.v0_id = 0
        self.v1_id = 0
        self.boundary = False

    @staticmethod
    cdef PyEdge from_cpp(Edge* e):
        cdef PyEdge pe = PyEdge.__new__(PyEdge)
        cdef unsigned i

        pe.id       = e.id()
        pe.length   = e.length()
        pe.v0_id    = e.v0().id()
        pe.v1_id    = e.v1().id()
        pe.boundary = e.is_boundary()
        return pe

    @property
    def vertex_ids(self):
        """Return (v0_id, v1_id) tuple."""
        return (self.v0_id, self.v1_id)

    def __repr__(self):
        return (
            f"PyEdge(id={self.id}, v0={self.v0_id}, v1={self.v1_id}, "
            f"length={self.length:.6g}, boundary={self.boundary})"
        )


# =============================================================================
# PyFace
# =============================================================================

cdef class PyFace:
    """Python-side snapshot of a geodesic::Face."""

    cdef readonly unsigned id
    cdef readonly tuple corner_angles

    def __cinit__(self):
        self.id = 0
        self.corner_angles = (0.0, 0.0, 0.0)

    @staticmethod
    cdef PyFace from_cpp(Face* f):
        cdef PyFace pf = PyFace.__new__(PyFace)
        cdef unsigned i
        cdef double* angles = f.corner_angles()

        pf.id = f.id()
        pf.corner_angles = (angles[0], angles[1], angles[2])
        return pf

    def __repr__(self):
        return (
            f"PyFace(id={self.id}, "
            f"vertices={self.adjacent_vertex_ids}, "
            f"edges={self.adjacent_edge_ids}, "
            f"angles=({self.corner_angles[0]:.4f}, "
            f"{self.corner_angles[1]:.4f}, "
            f"{self.corner_angles[2]:.4f}))"
        )


# =============================================================================
# PySurfacePoint
# =============================================================================

cdef class PySurfacePoint:
    """Python-side snapshot of a geodesic::SurfacePoint."""

    cdef double _x, _y, _z
    cdef unsigned _base_element_id
    cdef int _point_type

    def __cinit__(self):
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._base_element_id = 0
        self._point_type = UNDEFINED_POINT

    @staticmethod
    cdef PySurfacePoint from_cpp(SurfacePoint& sp):
        cdef PySurfacePoint p = PySurfacePoint.__new__(PySurfacePoint)
        p._x = sp.x()
        p._y = sp.y()
        p._z = sp.z()
        p._base_element_id = sp.base_element_id()
        p._point_type = <int>sp.type()
        return p

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def coords(self):
        return (self._x, self._y, self._z)

    @property
    def base_element_id(self):
        return self._base_element_id

    @property
    def point_type(self):
        return self._point_type

    @property
    def point_type_name(self):
        return _POINT_TYPE_NAMES.get(self._point_type, "UNKNOWN")

    @property
    def is_vertex(self):
        return self._point_type == VERTEX

    @property
    def is_edge(self):
        return self._point_type == EDGE

    @property
    def is_face(self):
        return self._point_type == FACE

    def __repr__(self):
        return (
            f"PySurfacePoint(type={self.point_type_name}, "
            f"id={self._base_element_id}, "
            f"coords=({self._x:.6g}, {self._y:.6g}, {self._z:.6g}))"
        )


# =============================================================================
# PyGeodesicAlgorithmExact
# =============================================================================

cdef class PyGeodesicAlgorithmExact:
    """
    Wrapper around C++ class GeodesicAlgorithmExact with exposed mesh data.
    """

    cdef GeodesicAlgorithmExact *algorithm
    cdef Mesh mesh
    cdef vector[double] points
    cdef vector[unsigned] faces
    cdef cbool _initialized
    cdef unsigned _num_edges
    cdef unsigned _num_vertices
    cdef unsigned _num_faces

    def __cinit__(self, _points, _faces):
        self._initialized = False

        try:
            _points = numpy.asarray(_points, dtype=numpy.float64)
            _faces = numpy.asarray(_faces, dtype=numpy.int32)
            assert len(_points.shape) == 2 and _points.shape[-1] == 3, \
                "'points' array has incorrect shape"
            assert len(_faces.shape) == 2 and _faces.shape[-1] == 3, \
                "'faces' array has incorrect shape"
            assert _faces.min() == 0 and _faces.max() == _points.shape[0] - 1, \
                'mesh vertices are not numbered sequentially from 0'
        except Exception as e:
            print(f'Error in PyGeodesicAlgorithmExact.__cinit__: {e}')
            self.algorithm = NULL
            return

        cdef numpy.float64_t coord
        for coord in _points.flatten():
            self.points.push_back(coord)

        cdef numpy.int32_t indx
        for indx in _faces.flatten():
            self.faces.push_back(indx)

        self.mesh.initialize_mesh_data(self.points, self.faces)
        self.algorithm = new GeodesicAlgorithmExact(&self.mesh)
        self._initialized = True

        self._num_vertices = self.mesh.vertices().size()
        self._num_faces = self.mesh.faces().size()
        self._num_edges = self.mesh.edges().size()

    # =========================================================================
    # Mesh element retrieval  (PyVertex / PyEdge / PyFace)
    # =========================================================================

    def getVertex(self, unsigned idx):
        """Return a PyVertex snapshot for vertex *idx*."""
        if idx >= self._num_vertices:
            raise IndexError(f"vertex index {idx} out of range [0, {self._num_vertices})")
        return PyVertex.from_cpp(&self.mesh.vertices()[idx], &self.mesh)

    def getVertices(self):
        """Return a list of all PyVertex objects."""
        cdef unsigned i
        result = []
        for i in range(self._num_vertices):
            result.append(PyVertex.from_cpp(&self.mesh.vertices()[i], &self.mesh))
        return result

    def getVerticesAsArray(self):
        """Return vertex coordinates as an Nx3 numpy array (fast path)."""
        cdef unsigned i
        cdef numpy.ndarray[numpy.float64_t, ndim=2] arr = numpy.empty(
            (self._num_vertices, 3), dtype=numpy.float64)
        for i in range(self._num_vertices):
            arr[i, 0] = self.mesh.vertices()[i].x()
            arr[i, 1] = self.mesh.vertices()[i].y()
            arr[i, 2] = self.mesh.vertices()[i].z()
        return arr

    def getEdge(self, unsigned idx):
        """Return a PyEdge snapshot for edge *idx*."""
        if idx >= self._num_edges:
            raise IndexError(f"edge index {idx} out of range [0, {self._num_edges})")
        return PyEdge.from_cpp(&self.mesh.edges()[idx])

    def getEdges(self):
        """Return a list of all PyEdge objects."""
        cdef unsigned i
        result = []
        for i in range(self._num_edges):
            result.append(PyEdge.from_cpp(&self.mesh.edges()[i]))
        return result

    def getFace(self, unsigned idx):
        """Return a PyFace snapshot for face *idx*."""
        if idx >= self._num_faces:
            raise IndexError(f"face index {idx} out of range [0, {self._num_faces})")
        return PyFace.from_cpp(&self.mesh.faces()[idx])

    def getFaces(self):
        """Return a list of all PyFace objects."""
        cdef unsigned i
        result = []
        for i in range(self._num_faces):
            result.append(PyFace.from_cpp(&self.mesh.faces()[i]))
        return result

    # =========================================================================
    # Mesh scalar queries (kept for backward compatibility)
    # =========================================================================

    def getEdgeCount(self):
        return self._num_edges

    def getVertexCount(self):
        return self._num_vertices

    def getFaceCount(self):
        return self._num_faces

    def getEdgeLengths(self):
        """Return current edge lengths as numpy array."""
        cdef unsigned i
        lengths = numpy.zeros(self._num_edges, dtype=numpy.float64)
        for i in range(self._num_edges):
            lengths[i] = self.mesh.edges()[i].length()
        return lengths

    def getEdgeVertices(self):
        """Return edge vertex indices as Nx2 numpy array."""
        cdef unsigned i
        edge_verts = numpy.zeros((self._num_edges, 2), dtype=numpy.uint32)
        for i in range(self._num_edges):
            edge_verts[i, 0] = self.mesh.edges()[i].v0().id()
            edge_verts[i, 1] = self.mesh.edges()[i].v1().id()
        return edge_verts

    def getFaceVertices(self, unsigned fid):
        cdef unsigned ids[3]
        get_face_vertex_ids(&self.mesh, fid, ids)
        return [ids[0], ids[1], ids[2]]

    def setEdgeLength(self, unsigned edge_idx, double new_length):
        set_edge_length(&self.mesh, edge_idx, new_length)

    def setEdgeLengthsFromArray(self, lengths):
        cdef unsigned i
        lengths = numpy.asarray(lengths, dtype=numpy.float64)
        if len(lengths) != self._num_edges:
            raise ValueError(
                f"Expected {self._num_edges} lengths, got {len(lengths)}")
        for i in range(self._num_edges):
            set_edge_length(&self.mesh, i, lengths[i])

    def getFaceEdgeLengths(self, unsigned face_idx):
        cdef double[3] lengths
        if face_idx >= self._num_faces:
            return None
        get_face_edge_lengths(&self.mesh, face_idx, lengths)
        return numpy.array([lengths[0], lengths[1], lengths[2]])

    def setCornerAngle(self, unsigned face_idx, unsigned corner_idx, double angle):
        set_corner_angle(&self.mesh, face_idx, corner_idx, angle)

    def getCornerAngles(self, unsigned face_idx):
        if face_idx >= self._num_faces:
            return None
        return numpy.array([
            get_corner_angle(&self.mesh, face_idx, 0),
            get_corner_angle(&self.mesh, face_idx, 1),
            get_corner_angle(&self.mesh, face_idx, 2)
        ])

    def setSaddleOrBoundary(self, unsigned vertex_idx, cbool value):
        set_saddle_or_boundary(&self.mesh, vertex_idx, value)

    def getSaddleOrBoundary(self, unsigned vertex_idx):
        if vertex_idx >= self._num_vertices:
            return None
        return get_saddle_or_boundary(&self.mesh, vertex_idx)

    def isEdgeBoundary(self, unsigned edge_idx):
        return is_edge_boundary(&self.mesh, edge_idx)

    def getEdgeBoundaryVertices(self, unsigned edge_idx):
        cdef unsigned v0, v1
        get_edge_vertices(&self.mesh, edge_idx, &v0, &v1)
        return (v0, v1)

    def rebuildAlgorithm(self):
        if self.algorithm != NULL:
            del self.algorithm
        self.algorithm = new GeodesicAlgorithmExact(&self.mesh)

    # =========================================================================
    # Geodesic Distance Methods
    # =========================================================================

    def geodesicDistance(self, sourceIndex, targetIndex):
        """
        Calculates the geodesic distance and path between two vertices.
        Returns (path_length, path_points_Nx3).
        """
        cdef Py_ssize_t i
        cdef vector[SurfacePoint] path

        def checkIndexWithinLimits(index):
            return index >= 0 and index <= self._num_vertices - 1

        try:
            assert self.algorithm != NULL, \
                "PyGeodesicAlgorithmExact class was not initialized correctly"
            sourceIndex = int(sourceIndex)
            targetIndex = int(targetIndex)
            assert checkIndexWithinLimits(sourceIndex), \
                "'sourceIndex' is outside limits of mesh"
            assert checkIndexWithinLimits(targetIndex), \
                "'targetIndex' is outside limits of mesh"
        except Exception as e:
            print(f'Error in PyGeodesicAlgorithmExact.geodesicDistance: {e}')
            return None, None

        cdef SurfacePoint source = SurfacePoint(
            &self.mesh.vertices()[sourceIndex])
        cdef SurfacePoint target = SurfacePoint(
            &self.mesh.vertices()[targetIndex])
        self.algorithm.geodesic(source, target, path)

        path_points = []
        for i in range(path.size()):
            path_points.append([path[i].x(), path[i].y(), path[i].z()])

        path_length = length(path)
        path_points = numpy.array(path_points)

        return path_length, path_points

    def checkIndicesWithinLimits(self, indices):
        return indices.min() >= 0 and indices.max() <= self._num_vertices - 1

    def propagate(self, source_indices, double max_distance=GEODESIC_INF):
        cdef vector[SurfacePoint] all_sources

        try:
            assert self.algorithm != NULL, \
                "PyGeodesicAlgorithmExact class was not initialized correctly"
            source_indices = numpy.asarray(source_indices, dtype=numpy.int32)
            assert len(source_indices.shape) == 1, \
                "'source_indices' array has incorrect shape"
            assert self.checkIndicesWithinLimits(source_indices), \
                "'source_indices' array outside limits of mesh"
        except Exception as e:
            print(f'Error in PyGeodesicAlgorithmExact.propagate: {e}')
            return

        for i in source_indices:
            all_sources.push_back(SurfacePoint(&self.mesh.vertices()[i]))

        self.algorithm.propagate(all_sources, max_distance)

    def geodesicDistances(self, source_indices=None, target_indices=None,
                          double max_distance=GEODESIC_INF):
        """
        Calculates the distance of each target vertex from the closest
        source vertex.
        """
        cdef Py_ssize_t i
        cdef numpy.int32_t indx
        cdef vector[SurfacePoint] all_sources
        cdef vector[SurfacePoint] stop_points
        cdef numpy.ndarray[numpy.float64_t, ndim=1] distances

        try:
            assert self.algorithm != NULL, \
                "PyGeodesicAlgorithmExact class was not initialized correctly"
            if source_indices is not None:
                source_indices = numpy.asarray(source_indices, dtype=numpy.int32)
                assert len(source_indices.shape) == 1, \
                    "'source_indices' array has incorrect shape"
                assert self.checkIndicesWithinLimits(source_indices), \
                    "'source_indices' array outside limits of mesh"
            if target_indices is not None:
                target_indices = numpy.asarray(target_indices, dtype=numpy.int32)
                assert len(target_indices.shape) == 1, \
                    "'target_indices' array has incorrect shape"
                assert self.checkIndicesWithinLimits(target_indices), \
                    "'target_indices' array outside limits of mesh"
        except Exception as e:
            print(f'Error in PyGeodesicAlgorithmExact.geodesicDistances: {e}')
            return None, None

        if source_indices is None:
            source_indices = numpy.array([0], dtype=numpy.int32)
        for i in source_indices:
            all_sources.push_back(SurfacePoint(&self.mesh.vertices()[i]))

        if target_indices is None:
            for i in range(self._num_vertices):
                stop_points.push_back(
                    SurfacePoint(&self.mesh.vertices()[i]))
            self.algorithm.propagate(all_sources, max_distance, NULL)
        else:
            for indx in target_indices:
                stop_points.push_back(
                    SurfacePoint(&self.mesh.vertices()[indx]))
            self.algorithm.propagate(all_sources, max_distance, &stop_points)

        distances = numpy.zeros((stop_points.size(),), dtype=numpy.float64)
        best_source = numpy.zeros((stop_points.size(),), dtype=numpy.int32)
        for i in range(stop_points.size()):
            best_source[i] = self.algorithm.best_source(
                stop_points[i], distances[i])
        distances[distances == GEODESIC_INF] = numpy.inf

        return distances, best_source

    def traceBack(self, unsigned target_idx):
        cdef SurfacePoint target
        cdef vector[SurfacePoint] path
        cdef Py_ssize_t i

        target = SurfacePoint(&self.mesh.vertices()[target_idx])
        self.algorithm.trace_back(target, path)

        result = []
        for i in range(<Py_ssize_t>path.size()):
            result.append(PySurfacePoint.from_cpp(path[i]))
        return result

    def __dealloc__(self):
        if self.algorithm != NULL:
            del self.algorithm


# =============================================================================
# Utility
# =============================================================================

def read_mesh_from_file(filename):
    """Read mesh from example files."""
    points = []
    faces = []
    with open(filename, 'r') as f:
        try:
            vals = f.readline().strip().split(' ')
            vals = [int(v) for v in vals]
            numpoints, numfaces = vals
            assert numpoints >= 3, "Number of points not >= 3"
        except AssertionError as e:
            print(f'Error reading header: {e}')
            return
        except Exception as e:
            print(f'Error reading header: {e}')
            return

        try:
            for i in range(numpoints):
                vals = f.readline().strip().replace('\t', ' ').split(' ')
                vals = [float(v) for v in vals]
                points.append(vals)
            points = numpy.array(points)
        except Exception as e:
            print(f'Error reading points: {e}')
            return

        try:
            for i in range(numfaces):
                vals = f.readline().strip().replace('\t', ' ').split(' ')
                vals = [int(v) for v in vals]
                faces.append(vals)
            faces = numpy.array(faces)
        except Exception as e:
            print(f'Error reading faces: {e}')
            return

    return points, faces

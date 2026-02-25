# distutils: language = c++
# -*- coding: utf-8 -*-
#!python
#cython: language_level=3

import numpy
cimport numpy
from libcpp.vector cimport vector
from libcpp cimport bool as cbool


cdef extern from "geodesic_kirsanov/geodesic_mesh_elements.h" namespace "geodesic":
    cdef cppclass Face:
        Face()

cdef extern from "geodesic_kirsanov/geodesic_mesh_elements.h" namespace "geodesic":
    cdef cppclass Vertex:
        Vertex()

cdef extern from "geodesic_kirsanov/geodesic_mesh_elements.h" namespace "geodesic":
    cdef cppclass SurfacePoint:
        SurfacePoint()
        SurfacePoint(Vertex*)
        double& x()
        double& y()
        double& z()

cdef extern from "geodesic_kirsanov/geodesic_mesh.h" namespace "geodesic":
    cdef cppclass Mesh:
        Mesh()
        void initialize_mesh_data(vector[double]&, vector[unsigned]&)
        vector[Vertex]& vertices()
        vector[Face]& faces()

cdef extern from "geodesic_kirsanov/geodesic_algorithm_exact.h" namespace "geodesic":
    cdef cppclass GeodesicAlgorithmExact:
        GeodesicAlgorithmExact(Mesh*)
        void propagate(vector[SurfacePoint]&, double, vector[SurfacePoint]*)
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
        """
        Initialize the geodesic algorithm with mesh data.
        """
        self._initialized = False
        
        try:
            _points = numpy.asarray(_points, dtype=numpy.float64)
            _faces = numpy.asarray(_faces, dtype=numpy.int32)
            assert len(_points.shape) == 2 and _points.shape[-1] == 3, "'points' array has incorrect shape"
            assert len(_faces.shape) == 2 and _faces.shape[-1] == 3, "'faces' array has incorrect shape"
            assert _faces.min() == 0 and _faces.max() == _points.shape[0] - 1, 'mesh vertices are not numbered sequentially from 0'
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
        
        # Cache counts
        self._num_vertices = self.mesh.vertices().size()
        self._num_faces = self.mesh.faces().size()
        # Compute number of edges (3 * faces / 2 for closed mesh, but use actual from mesh)
        # We'll compute this lazily when needed
        self._num_edges = 0

    def _computeEdgeCount(self):
        """Compute the number of edges by counting unique edges."""
        if self._num_edges == 0:
            # Count edges from face connectivity
            edges = set()
            for i in range(0, self.faces.size(), 3):
                v0 = self.faces[i]
                v1 = self.faces[i + 1]
                v2 = self.faces[i + 2]
                edges.add((min(v0, v1), max(v0, v1)))
                edges.add((min(v1, v2), max(v1, v2)))
                edges.add((min(v2, v0), max(v2, v0)))
            self._num_edges = len(edges)
        return self._num_edges

    # =========================================================================
    # Mesh Data Access Methods
    # =========================================================================
    
    def getEdgeCount(self):
        """Return the number of edges in the mesh."""
        return self._computeEdgeCount()

    def getVertexCount(self):
        """Return the number of vertices in the mesh."""
        return self._num_vertices

    def getFaceCount(self):
        """Return the number of faces in the mesh."""
        return self._num_faces

    def getEdgeLengths(self):
        """Return current edge lengths as numpy array."""
        cdef unsigned i
        cdef unsigned num_edges = self._computeEdgeCount()
        lengths = numpy.zeros(num_edges, dtype=numpy.float64)
        for i in range(num_edges):
            lengths[i] = get_edge_length(&self.mesh, i)
        return lengths

    def getEdgeVertices(self):
        """
        Return edge vertex indices as Nx2 numpy array.
        Each row contains (v0_id, v1_id) for the edge.
        Edges are sorted by (min_id, max_id).
        """
        cdef unsigned i
        cdef unsigned num_edges = self._computeEdgeCount()
        cdef unsigned v0, v1
        edge_verts = numpy.zeros((num_edges, 2), dtype=numpy.uint32)
        for i in range(num_edges):
            get_edge_vertices(&self.mesh, i, &v0, &v1)
            edge_verts[i, 0] = v0
            edge_verts[i, 1] = v1
        return edge_verts
    
    def getFaceVertices(self, unsigned fid):
        cdef unsigned ids[3]
        get_face_vertex_ids(&self.mesh,fid,ids)
        return [ids[0],ids[1],ids[2]]

    def setEdgeLength(self, unsigned edge_idx, double new_length):
        """Set the length of a single edge."""
        set_edge_length(&self.mesh, edge_idx, new_length)

    def setEdgeLengthsFromArray(self, lengths):
        """
        Set edge lengths from a numpy array.
        Array must have same length as number of edges.
        """
        cdef unsigned i
        cdef unsigned num_edges = self._computeEdgeCount()
        lengths = numpy.asarray(lengths, dtype=numpy.float64)
        
        if len(lengths) != num_edges:
            raise ValueError(f"Expected {num_edges} lengths, got {len(lengths)}")
        
        for i in range(num_edges):
            set_edge_length(&self.mesh, i, lengths[i])

    def getFaceEdgeLengths(self, unsigned face_idx):
        """Get the three edge lengths for a face (opposite to each vertex)."""
        cdef double[3] lengths
        
        if face_idx >= self._num_faces:
            return None
        
        get_face_edge_lengths(&self.mesh, face_idx, lengths)
        return numpy.array([lengths[0], lengths[1], lengths[2]])

    def setCornerAngle(self, unsigned face_idx, unsigned corner_idx, double angle):
        """Set corner angle for a face."""
        set_corner_angle(&self.mesh, face_idx, corner_idx, angle)

    def getCornerAngles(self, unsigned face_idx):
        """Get corner angles for a face."""
        if face_idx >= self._num_faces:
            return None
        return numpy.array([
            get_corner_angle(&self.mesh, face_idx, 0),
            get_corner_angle(&self.mesh, face_idx, 1),
            get_corner_angle(&self.mesh, face_idx, 2)
        ])

    def setSaddleOrBoundary(self, unsigned vertex_idx, cbool value):
        """Set saddle_or_boundary flag for a vertex."""
        set_saddle_or_boundary(&self.mesh, vertex_idx, value)

    def getSaddleOrBoundary(self, unsigned vertex_idx):
        """Get saddle_or_boundary flag for a vertex."""
        if vertex_idx >= self._num_vertices:
            return None
        return get_saddle_or_boundary(&self.mesh, vertex_idx)

    def isEdgeBoundary(self, unsigned edge_idx):
        """Check if an edge is a boundary edge."""
        return is_edge_boundary(&self.mesh, edge_idx)

    def getEdgeBoundaryVertices(self, unsigned edge_idx):
        """Get the vertex indices for an edge."""
        cdef unsigned v0, v1
        get_edge_vertices(&self.mesh, edge_idx, &v0, &v1)
        return (v0, v1)

    def rebuildAlgorithm(self):
        """Rebuild the geodesic algorithm after modifying edge lengths."""
        if self.algorithm != NULL:
            del self.algorithm
        self.algorithm = new GeodesicAlgorithmExact(&self.mesh)

    # =========================================================================
    # Geodesic Distance Methods
    # =========================================================================

    def geodesicDistance(self, sourceIndex, targetIndex):
        """
        Calculates the geodesic distance from the mesh vertex with index 'sourceIndex'
        to the mesh vertex with index 'targetIndex'.
        """
        cdef Py_ssize_t i
        cdef vector[SurfacePoint] path

        def checkIndexWithinLimits(index):
            return index >= 0 and index <= self._num_vertices - 1

        try:
            assert self.algorithm != NULL, "PyGeodesicAlgorithmExact class was not initialized correctly"
            sourceIndex = int(sourceIndex)
            targetIndex = int(targetIndex)
            assert checkIndexWithinLimits(sourceIndex), "'sourceIndex' is outside limits of mesh"
            assert checkIndexWithinLimits(targetIndex), "'targetIndex' is outside limits of mesh"
        except Exception as e:
            print(f'Error in PyGeodesicAlgorithmExact.geodesicDistance: {e}')
            return None, None

        cdef SurfacePoint source = SurfacePoint(&self.mesh.vertices()[sourceIndex])
        cdef SurfacePoint target = SurfacePoint(&self.mesh.vertices()[targetIndex])
        self.algorithm.geodesic(source, target, path)

        path_points = []
        for i in range(path.size()):
            path_points.append([path[i].x(), path[i].y(), path[i].z()])

        path_length = length(path)
        path_points = numpy.array(path_points)

        return path_length, path_points

    def geodesicDistances(self, source_indices=None, target_indices=None, double max_distance=GEODESIC_INF):
        """
        Calculates the distance of each target vertex from the best (closest) source vertex.
        """
        cdef Py_ssize_t i
        cdef numpy.int32_t indx
        cdef vector[SurfacePoint] all_sources
        cdef vector[SurfacePoint] stop_points
        cdef numpy.ndarray[numpy.float64_t, ndim=1] distances

        def checkIndicesWithinLimits(indices):
            return indices.min() >= 0 and indices.max() <= self._num_vertices - 1

        try:
            assert self.algorithm != NULL, "PyGeodesicAlgorithmExact class was not initialized correctly"
            if source_indices is not None:
                source_indices = numpy.asarray(source_indices, dtype=numpy.int32)
                assert len(source_indices.shape) == 1, "'source_indices' array has incorrect shape"
                assert checkIndicesWithinLimits(source_indices), "'source_indices' array outside limits of mesh"
            if target_indices is not None:
                target_indices = numpy.asarray(target_indices, dtype=numpy.int32)
                assert len(target_indices.shape) == 1, "'target_indices' array has incorrect shape"
                assert checkIndicesWithinLimits(target_indices), "'target_indices' array outside limits of mesh"
        except Exception as e:
            print(f'Error in PyGeodesicAlgorithmExact.geodesicDistances: {e}')
            return None, None

        if source_indices is None:
            source_indices = numpy.array([0], dtype=numpy.int32)
        for i in source_indices:
            all_sources.push_back(SurfacePoint(&self.mesh.vertices()[i]))

        if target_indices is None:
            for i in range(self._num_vertices):
                stop_points.push_back(SurfacePoint(&self.mesh.vertices()[i]))
            self.algorithm.propagate(all_sources, max_distance, NULL)
        else:
            for indx in target_indices:
                stop_points.push_back(SurfacePoint(&self.mesh.vertices()[indx]))
            self.algorithm.propagate(all_sources, max_distance, &stop_points)

        distances = numpy.zeros((stop_points.size(),), dtype=numpy.float64)
        best_source = numpy.zeros((stop_points.size(),), dtype=numpy.int32)
        for i in range(stop_points.size()):
            best_source[i] = self.algorithm.best_source(stop_points[i], distances[i])
        distances[distances == GEODESIC_INF] = numpy.inf

        return distances, best_source

    def __dealloc__(self):
        if self.algorithm != NULL:
            del self.algorithm


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

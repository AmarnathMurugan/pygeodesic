// geodesic_helpers.h
// Helper functions for modifying geodesic mesh from Python
#ifndef GEODESIC_HELPERS_H
#define GEODESIC_HELPERS_H

#include "geodesic_mesh.h"
#include "geodesic_mesh_elements.h"

namespace geodesic {

// Helper function to set edge length
inline void set_edge_length(Mesh* mesh, unsigned edge_idx, double length) {
    if (edge_idx < mesh->edges().size()) {
        mesh->edges()[edge_idx].length() = length;
    }
}

// Helper function to set corner angle
inline void set_corner_angle(Mesh* mesh, unsigned face_idx, unsigned corner_idx, double angle) {
    if (face_idx < mesh->faces().size() && corner_idx < 3) {
        mesh->faces()[face_idx].corner_angles()[corner_idx] = angle;
    }
}

// Helper function to set saddle_or_boundary flag
inline void set_saddle_or_boundary(Mesh* mesh, unsigned vertex_idx, bool value) {
    if (vertex_idx < mesh->vertices().size()) {
        mesh->vertices()[vertex_idx].saddle_or_boundary() = value;
    }
}

// Helper function to get edge length
inline double get_edge_length(Mesh* mesh, unsigned edge_idx) {
    if (edge_idx < mesh->edges().size()) {
        return mesh->edges()[edge_idx].length();
    }
    return 0.0;
}

// Helper function to get corner angle
inline double get_corner_angle(Mesh* mesh, unsigned face_idx, unsigned corner_idx) {
    if (face_idx < mesh->faces().size() && corner_idx < 3) {
        return mesh->faces()[face_idx].corner_angles()[corner_idx];
    }
    return 0.0;
}

// Helper function to get saddle_or_boundary flag
inline bool get_saddle_or_boundary(Mesh* mesh, unsigned vertex_idx) {
    if (vertex_idx < mesh->vertices().size()) {
        return mesh->vertices()[vertex_idx].saddle_or_boundary();
    }
    return false;
}

// Helper to get edge vertex IDs
inline void get_edge_vertices(Mesh* mesh, unsigned edge_idx, unsigned* v0, unsigned* v1) {
    if (edge_idx < mesh->edges().size()) {
        *v0 = mesh->edges()[edge_idx].adjacent_vertices()[0]->id();
        *v1 = mesh->edges()[edge_idx].adjacent_vertices()[1]->id();
    }
}

// Helper to check if edge is boundary
inline bool is_edge_boundary(Mesh* mesh, unsigned edge_idx) {
    if (edge_idx < mesh->edges().size()) {
        return mesh->edges()[edge_idx].is_boundary();
    }
    return false;
}

// Helper to get face edge lengths (opposite to each vertex)
inline void get_face_edge_lengths(Mesh* mesh, unsigned face_idx, double* lengths) {
    if (face_idx < mesh->faces().size()) {
        Face& f = mesh->faces()[face_idx];
        for (unsigned j = 0; j < 3; ++j) {
            vertex_pointer v = f.adjacent_vertices()[j];
            lengths[j] = f.opposite_edge(v)->length();
        }
    }
}

inline void get_face_vertex_ids(Mesh* mesh, unsigned face_idx, unsigned* vertex_ids) {
    if (face_idx < mesh->faces().size()) {
        Face& f = mesh->faces()[face_idx];
        for (unsigned j = 0; j < 3; ++j) {
            vertex_pointer v = f.adjacent_vertices()[j];
            vertex_ids[j] = v->id();
        }
    }
}
} // namespace geodesic

#endif // GEODESIC_HELPERS_H

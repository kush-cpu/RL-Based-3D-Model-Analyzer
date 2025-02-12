import streamlit as st
import trimesh
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import io
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

class ModelQualityPredictor(nn.Module):
    def __init__(self):
        super(ModelQualityPredictor, self).__init__()
        # Input features: vertex count, face count, uv metrics, etc.
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

def extract_model_features(mesh):
    """Extract numerical features from the 3D model for ML analysis"""
    features = {
        'vertex_count': len(mesh.vertices),
        'face_count': len(mesh.faces),
        'volume': mesh.volume if hasattr(mesh, 'volume') else 0,
        'surface_area': mesh.area,
        'is_watertight': float(mesh.is_watertight),
        'is_winding_consistent': float(mesh.is_winding_consistent),
        'euler_number': len(mesh.vertices) - len(mesh.faces) + len(mesh.edges),
        'bbox_volume': np.prod(mesh.extents),
        'compactness': mesh.area ** (3/2) / (36 * np.pi * mesh.volume ** 2) if mesh.volume > 0 else 0,
        # Calculate edge lengths manually using vertex positions
        'edge_length_mean': calculate_edge_lengths(mesh)
    }
    return features

def calculate_edge_lengths(mesh):
    """Calculate mean edge length for a mesh"""
    try:
        # Get unique edges
        edges = mesh.edges_unique
        # Get vertices for each edge
        edge_vertices = mesh.vertices[edges]
        # Calculate edge lengths using numpy
        lengths = np.linalg.norm(edge_vertices[:, 1] - edge_vertices[:, 0], axis=1)
        return float(np.mean(lengths)) if len(lengths) > 0 else 0
    except:
        return 0

def analyze_uv_maps(mesh):
    """Analyze UV mapping quality and coverage"""
    if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv'):
        return {
            'has_uvs': False,
            'coverage': 0,
            'overlapping': False,
            'suggestions': ['Model lacks UV mapping. Consider adding UV coordinates.']
        }
    
    uv_coords = mesh.visual.uv
    
    # Calculate UV coverage
    uv_area = np.abs(np.cross(uv_coords[1:] - uv_coords[0], uv_coords[2:] - uv_coords[0])).sum() / 2
    coverage = min(uv_area / 1.0, 1.0)  # Normalize to [0,1]
    
    # Check for overlapping UVs (simplified)
    overlapping = len(np.unique(uv_coords, axis=0)) < len(uv_coords)
    
    # More detailed UV analysis
    uv_bounds = np.array([uv_coords.min(axis=0), uv_coords.max(axis=0)])
    uv_size = uv_bounds[1] - uv_bounds[0]
    
    suggestions = []
    if coverage < 0.8:
        suggestions.append('UV coverage is low. Consider repacking UVs to maximize texture space.')
    if overlapping:
        suggestions.append('Detected overlapping UVs. Consider unwrapping affected areas.')
    if np.any(uv_size > 1.0):
        suggestions.append('UVs extend beyond 0-1 space. Consider normalizing UV coordinates.')
    
    return {
        'has_uvs': True,
        'coverage': coverage,
        'overlapping': overlapping,
        'uv_size': uv_size.tolist(),
        'suggestions': suggestions
    }

def analyze_topology(mesh):
    """Analyze mesh topology and provide suggestions"""
    suggestions = []
    
    # Basic topology checks
    vertex_count = len(mesh.vertices)
    face_count = len(mesh.faces)
    
    if vertex_count > 10000:
        suggestions.append(f'High vertex count ({vertex_count:,}). Consider decimation for game optimization.')
    
    if hasattr(mesh, 'is_watertight') and not mesh.is_watertight:
        suggestions.append('Mesh has holes or non-manifold edges. Fix for proper game engine import.')
    
    # Calculate face density
    try:
        area = mesh.area
        density = face_count / area if area > 0 else 0
        if density > 1000:  # arbitrary threshold
            suggestions.append('High polygon density detected. Consider optimization.')
    except:
        density = 0
    
    # Check for separate components using connected components
    try:
        components = mesh.split() if hasattr(mesh, 'split') else [mesh]
        part_count = len(components)
        if part_count > 1:
            suggestions.append(f'Mesh consists of {part_count} separate parts. Consider unifying or organizing parts.')
    except:
        part_count = 1
    
    # Check for degenerate faces using face areas
    try:
        face_areas = mesh.area_faces
        degenerate_count = np.sum(face_areas < 1e-8)
        if degenerate_count > 0:
            suggestions.append(f'Found {degenerate_count} potentially degenerate faces. Clean up geometry.')
    except:
        degenerate_count = 0
    
    return {
        'vertex_count': vertex_count,
        'face_count': face_count,
        'is_watertight': mesh.is_watertight if hasattr(mesh, 'is_watertight') else False,
        'density': density,
        'part_count': part_count,
        'suggestions': suggestions
    }

def get_game_readiness_score(analysis_results):
    """Calculate overall game-readiness score based on various metrics"""
    score = 100
    
    # Topology penalties
    if analysis_results['topology']['vertex_count'] > 10000:
        score -= min(30, (analysis_results['topology']['vertex_count'] - 10000) / 1000)
    
    if not analysis_results['topology']['is_watertight']:
        score -= 15
    
    if analysis_results['topology']['part_count'] > 1:
        score -= min(10, analysis_results['topology']['part_count'] * 2)
    
    # UV penalties
    if not analysis_results['uv_analysis']['has_uvs']:
        score -= 30
    elif analysis_results['uv_analysis']['coverage'] < 0.8:
        score -= 20 * (1 - analysis_results['uv_analysis']['coverage'])
    
    if analysis_results['uv_analysis'].get('overlapping', False):
        score -= 15
    
    return max(0, min(100, score))

def main():
    st.title("3D Model Analyzer for Game Development")
    st.write("Upload your 3D model to get analysis and optimization suggestions")
    
    uploaded_file = st.file_uploader("Choose a 3D model file", type=['obj', 'stl', 'ply', 'glb', 'fbx'])
    
    if uploaded_file is not None:
        try:
            # Load the model directly from the uploaded file object
            mesh = trimesh.load(io.BytesIO(uploaded_file.getvalue()), file_type=uploaded_file.name.split('.')[-1])
            
            # Perform analysis
            uv_analysis = analyze_uv_maps(mesh)
            topology_analysis = analyze_topology(mesh)
            
            # Extract features for ML prediction
            features = extract_model_features(mesh)
            
            analysis_results = {
                'uv_analysis': uv_analysis,
                'topology': topology_analysis,
                'features': features
            }
            
            # Calculate game readiness score
            game_readiness = get_game_readiness_score(analysis_results)
            
            # Display results
            st.header("Analysis Results")
            
            # Display game readiness score with color coding
            score_color = 'green' if game_readiness >= 80 else 'orange' if game_readiness >= 60 else 'red'
            st.markdown(f"### Game Readiness Score: <span style='color:{score_color}'>{game_readiness:.1f}%</span>", unsafe_allow_html=True)
            
            # Detailed Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Topology Analysis")
                st.write(f"Vertex Count: {topology_analysis['vertex_count']:,}")
                st.write(f"Face Count: {topology_analysis['face_count']:,}")
                st.write(f"Watertight Mesh: {'Yes' if topology_analysis['is_watertight'] else 'No'}")
                st.write(f"Number of Parts: {topology_analysis['part_count']}")
                if topology_analysis['density'] > 0:
                    st.write(f"Polygon Density: {topology_analysis['density']:.2f} faces/unit²")
            
            with col2:
                st.subheader("UV Analysis")
                if uv_analysis['has_uvs']:
                    st.write(f"UV Coverage: {uv_analysis['coverage']*100:.1f}%")
                    st.write(f"Overlapping UVs: {'Yes' if uv_analysis['overlapping'] else 'No'}")
                    st.write("UV Size:")
                    st.write(f"- Width: {uv_analysis['uv_size'][0]:.2f}")
                    st.write(f"- Height: {uv_analysis['uv_size'][1]:.2f}")
                else:
                    st.write("No UV mapping found")
            
            # Suggestions
            st.subheader("Optimization Suggestions")
            all_suggestions = (
                topology_analysis['suggestions'] +
                uv_analysis['suggestions']
            )
            for suggestion in all_suggestions:
                st.write(f"• {suggestion}")
            
        except Exception as e:
            st.error(f"Error processing the model: {str(e)}")

if __name__ == "__main__":
    main()